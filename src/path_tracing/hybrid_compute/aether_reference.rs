//! PROMETHEUS-owned, acceptance-only stochastic spectral atmosphere reference.
//!
//! The production AETHER path is LUT based.  This deliberately independent
//! reference dispatches `main_aether_spectral_reference` through
//! [`HybridPathTracer`], reusing PROMETHEUS camera rays, RNG, heightfield
//! traversal and shadow visibility while binding no AETHER LUT or environment.

use super::terrain_heightfield::TerrainPtScene;
use super::*;
use crate::core::memory_tracker::global_tracker;
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};

/// Inputs for the dedicated AETHER acceptance reference.
#[derive(Clone)]
pub struct AetherSpectralReferenceDesc {
    pub heights: Vec<f32>,
    pub dem_width: u32,
    pub dem_height: u32,
    pub spacing: (f32, f32),
    pub exaggeration: f32,
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub turbidity: f32,
    pub ozone_du: f32,
    pub mie_g: f32,
    pub ground_albedo: f32,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    pub spp: u32,
    pub enabled: bool,
    pub variance_threshold: f32,
    /// Timestamp queries are needed only when the caller requested a render
    /// certificate. Acceptance sweeps deliberately leave them off so repeated
    /// stochastic dispatches do not allocate and tear down query pools.
    pub collect_timing: bool,
}

/// Linear, untonemapped stochastic reference and its convergence evidence.
pub struct AetherSpectralReferenceOutput {
    /// Unclipped per-pixel mean CIE XYZ. Acceptance callers can combine
    /// independent batches before the one signed-RGB conversion and clip.
    pub mean_xyz: Vec<f32>,
    pub linear_rgb: Vec<f32>,
    /// Maximum per-pixel estimated variance of the unbiased sample mean CIE Y.
    pub variance: f32,
    pub converged: bool,
    pub seed: u32,
    pub spp: u32,
    /// Sum of camera samples whose exact primary ray hit PROMETHEUS terrain.
    pub terrain_primary_hits: u64,
    pub gpu_resource_bytes: u64,
}

fn finite3(value: [f32; 3]) -> bool {
    value.iter().all(|component| component.is_finite())
}

fn aether_xyz_to_signed_linear_rgb(xyz: [f32; 3]) -> [f32; 3] {
    [
        (3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2]) / 3.2613921,
        (-0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2]) / 2.5069624,
        (0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2]) / 2.3679786,
    ]
}

fn aether_finalize_xyz_sum(sum_xyz: [f32; 3], sample_count: u32) -> [f32; 3] {
    debug_assert!(sample_count > 0);
    let inverse_count = 1.0 / sample_count as f32;
    let mean_xyz = sum_xyz.map(|component| component * inverse_count);
    // This is the estimator's only non-negative clip: averaging remains linear
    // in XYZ, including components that map to negative signed linear RGB.
    aether_xyz_to_signed_linear_rgb(mean_xyz).map(|component| component.max(0.0))
}

fn validate_desc(desc: &AetherSpectralReferenceDesc) -> Result<(), RenderError> {
    let invalid = |message: &str| Err(RenderError::Render(message.into()));
    if desc.width == 0 || desc.height == 0 {
        return invalid("AETHER spectral reference requires non-zero width and height");
    }
    if desc.spp == 0 || desc.spp > 4096 {
        return invalid("AETHER spectral reference spp must be in 1..=4096");
    }
    let spectral_paths = (desc.width as u64)
        .saturating_mul(desc.height as u64)
        .saturating_mul(desc.spp as u64)
        .saturating_mul(crate::core::atmosphere::NUM_WAVELENGTHS as u64);
    if spectral_paths > 8_000_000 {
        return Err(RenderError::Render(format!(
            "AETHER spectral reference request has {spectral_paths} wavelength paths; acceptance lane limit is 8000000"
        )));
    }
    if !(desc.spacing.0.is_finite()
        && desc.spacing.0 > 0.0
        && desc.spacing.1.is_finite()
        && desc.spacing.1 > 0.0)
    {
        return invalid("AETHER spectral reference spacing must be finite and positive");
    }
    if !(desc.exaggeration.is_finite() && desc.exaggeration > 0.0) {
        return invalid("AETHER spectral reference exaggeration must be finite and positive");
    }
    if !(finite3(desc.cam_origin) && finite3(desc.cam_look_at) && finite3(desc.cam_up)) {
        return invalid("AETHER spectral reference camera vectors must be finite");
    }
    let origin = glam::Vec3::from(desc.cam_origin);
    let forward = glam::Vec3::from(desc.cam_look_at) - origin;
    if forward.length() < 1e-6
        || forward
            .normalize()
            .cross(glam::Vec3::from(desc.cam_up))
            .length()
            < 1e-6
    {
        return invalid("AETHER spectral reference camera basis is degenerate");
    }
    let planet_center = glam::Vec3::new(0.0, -6_360_000.0, 0.0);
    let observer_altitude = (origin - planet_center).length() - 6_360_000.0;
    if !(observer_altitude >= 0.0 && observer_altitude < 100_000.0) {
        return invalid("AETHER spectral reference camera must be inside the 0..100 km atmosphere");
    }
    if !(desc.fov_y_deg.is_finite() && desc.fov_y_deg > 0.0 && desc.fov_y_deg < 180.0) {
        return invalid("AETHER spectral reference fov_y_deg must be in (0, 180)");
    }
    if !(desc.sun_azimuth_deg.is_finite()
        && desc.sun_elevation_deg.is_finite()
        && desc.sun_intensity.is_finite()
        && desc.sun_intensity >= 0.0)
    {
        return invalid(
            "AETHER spectral reference sun inputs must be finite and intensity non-negative",
        );
    }
    let mut atmosphere = crate::core::atmosphere::AtmosphereConfig::default();
    atmosphere.turbidity = desc.turbidity;
    atmosphere.ozone_du = desc.ozone_du;
    atmosphere.mie_g = desc.mie_g;
    atmosphere.ground_albedo = desc.ground_albedo;
    atmosphere.validate().map_err(|error| {
        RenderError::Render(format!(
            "invalid canonical AETHER settings for spectral reference: {error}"
        ))
    })?;
    if !(desc.variance_threshold.is_finite() && desc.variance_threshold > 0.0) {
        return invalid("AETHER spectral reference variance_threshold must be finite and positive");
    }
    Ok(())
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, RenderError> {
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("aether-spectral-reference-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("aether-spectral-reference-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(bytes)
}

impl HybridPathTracer {
    /// Dispatch the genuine stochastic spectral reference.  This API is
    /// intentionally separate from the production LUT renderer.
    pub fn render_aether_spectral_reference(
        &self,
        desc: &AetherSpectralReferenceDesc,
    ) -> Result<AetherSpectralReferenceOutput, RenderError> {
        validate_desc(desc)?;
        let context = try_ctx()?;
        let device = &context.device;
        let queue = &context.queue;
        let terrain_scene = TerrainPtScene::new(
            device,
            queue,
            &desc.heights,
            desc.dem_width,
            desc.dem_height,
            desc.spacing,
            desc.exaggeration,
            [desc.ground_albedo; 3],
            None,
            0.0,
        )?;
        let hybrid_scene = HybridScene::new();
        let origin = glam::Vec3::from(desc.cam_origin);
        let forward = (glam::Vec3::from(desc.cam_look_at) - origin).normalize();
        let right = forward.cross(glam::Vec3::from(desc.cam_up)).normalize();
        let up = right.cross(forward).normalize();
        let azimuth = desc.sun_azimuth_deg.to_radians();
        let elevation = desc.sun_elevation_deg.to_radians();
        let sun_direction = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        ];

        let base = Uniforms {
            width: desc.width,
            height: desc.height,
            frame_index: 0,
            // This entry owns no conventional AOVs; use the otherwise-idle
            // field as its explicit dispatch enable bit rather than hiding an
            // atmosphere flag inside PROMETHEUS's terrain feature mask.
            aov_flags: u32::from(desc.enabled),
            cam_origin: desc.cam_origin,
            cam_fov_y: desc.fov_y_deg.to_radians(),
            cam_right: right.into(),
            cam_aspect: desc.width as f32 / desc.height as f32,
            cam_up: up.into(),
            cam_exposure: 1.0,
            cam_forward: forward.into(),
            seed_hi: desc.seed,
            // A plain `seed ^ constant` would cancel against seed_hi in the
            // shader's XOR stream initializer. Rotate before mixing so the
            // public seed genuinely selects a stochastic sequence.
            seed_lo: desc.seed.rotate_left(16) ^ 0x85EB_CA6B,
            camera_model: 0,
            full_width: desc.width,
            full_height: desc.height,
            pixel_offset_x: 0,
            pixel_offset_y: 0,
            ortho_half_height: 1.0,
            camera_flags: 0,
            sensor_rect: [0.0, 0.0, 1.0, 1.0],
        };
        let lighting = LightingUniforms {
            light_dir: sun_direction,
            lighting_type: 1,
            light_color: [desc.sun_intensity; 3],
            shadows_enabled: 1,
            ambient_color: [0.0; 3],
            shadow_intensity: 1.0,
            hdri_intensity: 0.0,
            hdri_rotation: 0.0,
            specular_power: 0.0,
            _pad: [0; 5],
        };
        let hybrid = HybridUniforms {
            sdf_primitive_count: 0,
            sdf_node_count: 0,
            mesh_vertex_count: 0,
            mesh_index_count: 0,
            mesh_bvh_node_count: 0,
            traversal_mode: TraversalMode::TerrainOnly as u32,
            _pad: [0; 2],
        };
        let mut terrain = terrain_scene.uniforms(desc.spp, 2);
        terrain.mips[1] = 1;
        terrain.h_params[3] = desc.ozone_du / 300.0;
        terrain.albedo_pad[3] = desc.ground_albedo;
        terrain.extra[2] = desc.turbidity.to_bits();
        terrain.extra[3] = desc.mie_g.to_bits();

        let base_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("aether-spectral-reference-base-ubo"),
                contents: bytemuck::bytes_of(&base),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let lighting_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("aether-spectral-reference-lighting-ubo"),
                contents: bytemuck::bytes_of(&lighting),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let hybrid_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("aether-spectral-reference-hybrid-ubo"),
                contents: bytemuck::bytes_of(&hybrid),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let terrain_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("aether-spectral-reference-terrain-ubo"),
                contents: bytemuck::bytes_of(&terrain),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let earth_curvature =
            <super::terrain_heightfield::EarthCurvatureUniforms as bytemuck::Zeroable>::zeroed();
        let earth_curvature_ubo = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("aether-spectral-reference-earth-curvature-ubo"),
                contents: bytemuck::bytes_of(&earth_curvature),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let scene_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("aether-spectral-reference-scene-dummy"),
                size: std::mem::size_of::<Sphere>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let pixel_count = (desc.width as u64) * (desc.height as u64);
        let accum_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("aether-spectral-reference-accum"),
                size: pixel_count * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let welford_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("aether-spectral-reference-welford"),
                size: pixel_count * 8,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let reservoir_stride = std::mem::size_of::<crate::path_tracing::restir::Reservoir>() as u64;
        let reservoir_curr = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("aether-spectral-reference-reservoir-curr-dummy"),
                size: reservoir_stride,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let reservoir_prev = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("aether-spectral-reference-reservoir-prev-dummy"),
                size: reservoir_stride,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;

        let metrics = global_tracker().get_metrics();
        if metrics.total_bytes > metrics.limit_bytes
            || metrics.host_visible_bytes > metrics.limit_bytes
        {
            return Err(RenderError::Budget(format!(
                    "AETHER spectral reference exceeds memory budget: total={} host_visible={} limit={}",
                    metrics.total_bytes, metrics.host_visible_bytes, metrics.limit_bytes
                )));
        }

        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aether-spectral-reference-bg0"),
            layout: &self.layouts.uniforms,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_ubo.as_entire_binding(),
                },
            ],
        });
        let mut group1_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hybrid_ubo.as_entire_binding(),
            },
        ];
        group1_entries.extend(
            hybrid_scene
                .get_mesh_bind_entries()?
                .into_iter()
                .enumerate()
                .map(|(index, mut entry)| {
                    entry.binding = index as u32 + 2;
                    entry
                }),
        );
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aether-spectral-reference-bg1"),
            layout: &self.layouts.scene,
            entries: &group1_entries,
        });
        let height_view = terrain_scene
            .pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = terrain_scene
            .pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let env_view = terrain_scene
            .env_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let albedo_view = terrain_scene
            .albedo_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aether-spectral-reference-bg2"),
            layout: &self.layouts.accum,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&minmax_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: welford_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: reservoir_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: reservoir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: earth_curvature_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
            ],
        });

        let mut timing = desc.collect_timing.then(|| {
            crate::core::gpu_timing::OneShotTiming::for_device(
                context.device.clone(),
                context.queue.clone(),
            )
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aether-spectral-reference-encoder"),
        });
        let timing_scope = timing
            .as_mut()
            .and_then(|timing| timing.begin(&mut encoder, "hybrid_pt.aether_spectral_reference"));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aether-spectral-reference-pass"),
                ..Default::default()
            });
            // setup.rs registers this exact assembled source under the
            // canonical hybrid label; certificate provenance must use it.
            crate::core::shader_registry::record_shader_use("hybrid-pt-kernel");
            pass.set_pipeline(&self.pipeline_aether_reference);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &group2, &[]);
            pass.dispatch_workgroups(desc.width.div_ceil(8), desc.height.div_ceil(8), 1);
        }
        if let Some(timing) = timing.as_mut() {
            timing.end(&mut encoder, timing_scope, 1);
            timing.resolve(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);
        if let Some(timing) = timing {
            if !timing.record_into_certificate() {
                crate::core::certificate::record_pass(
                    "hybrid_pt.aether_spectral_reference",
                    0.0,
                    1,
                );
            }
        } else {
            // `certificate=` controls emission, but CENSOR keeps the last
            // capture inspectable even when emission is off. State the
            // repeatability trade explicitly instead of presenting the
            // zero fallback as a live timestamp.
            crate::core::degradation::record_degradation(
                "timing_unavailable",
                "hybrid_pt.aether_spectral_reference",
                "timestamp collection disabled for the repeated stochastic acceptance sweep",
            );
            crate::core::certificate::record_pass("hybrid_pt.aether_spectral_reference", 0.0, 1);
        }

        let accum_bytes = read_buffer(device, queue, &accum_buffer, pixel_count * 16)?;
        let welford_bytes = read_buffer(device, queue, &welford_buffer, pixel_count * 8)?;
        let accum: &[f32] = bytemuck::cast_slice(&accum_bytes);
        let welford: &[f32] = bytemuck::cast_slice(&welford_bytes);
        if accum
            .iter()
            .chain(welford.iter())
            .any(|value| !value.is_finite())
        {
            return Err(RenderError::Render(
                "AETHER spectral reference produced non-finite transport output".into(),
            ));
        }
        let mut mean_xyz = Vec::with_capacity(pixel_count as usize * 3);
        let mut linear_rgb = Vec::with_capacity(pixel_count as usize * 3);
        let mut terrain_primary_hits = 0u64;
        for pixel in accum.chunks_exact(4) {
            let sum_xyz = [pixel[0], pixel[1], pixel[2]];
            let inverse_count = 1.0 / desc.spp as f32;
            mean_xyz.extend(sum_xyz.map(|component| component * inverse_count));
            let final_rgb = aether_finalize_xyz_sum(sum_xyz, desc.spp);
            linear_rgb.extend_from_slice(&final_rgb);
            terrain_primary_hits += pixel[3].round().max(0.0) as u64;
        }
        let variance = if desc.enabled && desc.spp > 1 {
            let denominator = desc.spp as f32 * (desc.spp - 1) as f32;
            welford
                .chunks_exact(2)
                .map(|pixel| pixel[1] / denominator)
                .fold(0.0f32, f32::max)
        } else {
            0.0
        };
        let gpu_resource_bytes = terrain_scene.byte_size()
            + base_ubo.size()
            + lighting_ubo.size()
            + hybrid_ubo.size()
            + terrain_ubo.size()
            + scene_buffer.size()
            + accum_buffer.size()
            + welford_buffer.size()
            + reservoir_curr.size()
            + reservoir_prev.size();

        Ok(AetherSpectralReferenceOutput {
            mean_xyz,
            linear_rgb,
            variance,
            converged: !desc.enabled || (desc.spp > 1 && variance <= desc.variance_threshold),
            seed: desc.seed,
            spp: desc.spp,
            terrain_primary_hits,
            gpu_resource_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_is_a_real_prometheus_spectral_transport() {
        let source = include_str!("../../shaders/atmosphere/prometheus_spectral_reference.wgsl");
        assert!(source.contains("fn main_aether_spectral_reference"));
        assert!(source.contains("AETHER_REF_WAVELENGTH_COUNT: u32 = 11u"));
        assert!(source.contains("aether_ref_trace_wavelength"));
        assert!(source.contains("free_flight"));
        assert!(source.contains("aether_ref_sample_rayleigh"));
        assert!(source.contains("aether_ref_sample_mie"));
        assert!(source.contains("AETHER_REF_MAX_DEPTH: u32 = 6u"));
        assert!(source.contains("AETHER_REF_RR_START_DEPTH"));
        assert!(source.contains("AETHER_REF_PLANET_RAY_OFFSET_M: f32 = 2.0"));
        assert!(source.contains("fn aether_ref_surface_ray_origin"));
        assert!(source
            .contains("normal * (AETHER_REF_BOTTOM_RADIUS_M + AETHER_REF_PLANET_RAY_OFFSET_M)"));
        assert!(source.contains("Ray(surface_ray_origin, 1e-3, bounce_direction, 1e30)"));
        assert!(source.contains("intersect_hybrid(camera_ray)"));
        assert!(source.contains("intersect_shadow_ray(shadow_ray, top_t)"));
        assert!(!source.contains("textureSample"));
        assert!(!source.contains("sample_inscatter"));
        assert!(!source.contains("terrain_env_radiance"));
        assert!(source.contains("sum_xyz = sum_xyz + xyz"));
        assert!(source.contains("let sample_y = xyz.y"));
        assert!(!source.contains("sample_rgb = max"));
        assert!(!source.contains("aether_ref_xyz_to_rgb"));
        let driver = include_str!("aether_reference.rs");
        assert!(driver.contains("OneShotTiming::for_device"));
        assert!(driver.contains("hybrid_pt.aether_spectral_reference"));
        assert!(driver.contains("record_shader_use(\"hybrid-pt-kernel\")"));
        let python_driver = include_str!("../../py_functions/path_tracing/aether_reference.rs");
        assert!(python_driver.contains("AETHER_REFERENCE_TRACER"));
        assert!(python_driver.contains("get_or_try_init(HybridPathTracer::new)"));
    }

    #[test]
    fn scale_safe_planet_ground_origins_do_not_self_intersect() {
        const RADIUS: f32 = 6_360_000.0;
        const OFFSET: f32 = 2.0;
        let center = glam::Vec3::new(0.0, -RADIUS, 0.0);
        let positive_ground_root = |origin: glam::Vec3, direction: glam::Vec3| {
            let oc = origin - center;
            let b = oc.dot(direction);
            let c = oc.dot(oc) - RADIUS * RADIUS;
            let discriminant = b * b - c;
            if discriminant < 0.0 {
                return None;
            }
            let root = discriminant.sqrt();
            [-b - root, -b + root]
                .into_iter()
                .find(|distance| *distance > 1.0e-3)
        };

        for normal in [
            glam::Vec3::Y,
            glam::Vec3::new(1.0, 1.0, 0.0).normalize(),
            glam::Vec3::new(0.5, 0.5, std::f32::consts::FRAC_1_SQRT_2).normalize(),
        ] {
            let origin = center + normal * (RADIUS + OFFSET);
            let tangent = if normal.y.abs() < 0.9 {
                normal.cross(glam::Vec3::Y).normalize()
            } else {
                normal.cross(glam::Vec3::X).normalize()
            };
            for cosine in [89.0_f32.to_radians().sin(), 0.5, 0.1] {
                let direction =
                    (normal * cosine + tangent * (1.0 - cosine * cosine).sqrt()).normalize();
                assert!(
                    positive_ground_root(origin, direction).is_none(),
                    "normal={normal:?}, direction={direction:?}"
                );
            }
        }
    }

    #[test]
    fn final_conversion_commutes_with_partitioned_xyz_accumulation() {
        // These two non-negative XYZ samples map to opposite signed RGB
        // components. Clipping either partition before recombination is biased.
        let first = [0.0, 1.0, 0.0];
        let second = [1.0, 0.0, 0.0];
        let whole_sum = [
            first[0] + second[0],
            first[1] + second[1],
            first[2] + second[2],
        ];
        let split_sum = [
            [first[0], second[0]].into_iter().sum(),
            [first[1], second[1]].into_iter().sum(),
            [first[2], second[2]].into_iter().sum(),
        ];
        let whole = aether_finalize_xyz_sum(whole_sum, 2);
        let recombined = aether_finalize_xyz_sum(split_sum, 2);
        assert_eq!(whole, recombined);

        let clipped_first = aether_finalize_xyz_sum(first, 1);
        let clipped_second = aether_finalize_xyz_sum(second, 1);
        let prematurely_clipped = [
            0.5 * (clipped_first[0] + clipped_second[0]),
            0.5 * (clipped_first[1] + clipped_second[1]),
            0.5 * (clipped_first[2] + clipped_second[2]),
        ];
        assert!(whole
            .iter()
            .zip(prematurely_clipped)
            .any(|(unbiased, biased)| (unbiased - biased).abs() > 1.0e-3));
    }

    #[test]
    fn rejects_a_camera_outside_the_atmosphere() {
        let desc = AetherSpectralReferenceDesc {
            heights: vec![0.0; 4],
            dem_width: 2,
            dem_height: 2,
            spacing: (1.0, 1.0),
            exaggeration: 1.0,
            cam_origin: [0.0, 100_001.0, 0.0],
            cam_look_at: [1.0, 100_001.0, 0.0],
            cam_up: [0.0, 1.0, 0.0],
            fov_y_deg: 45.0,
            sun_azimuth_deg: 90.0,
            sun_elevation_deg: 10.0,
            sun_intensity: 1.0,
            turbidity: 2.0,
            ozone_du: 300.0,
            mie_g: 0.8,
            ground_albedo: 0.3,
            width: 1,
            height: 1,
            seed: 7,
            spp: 1,
            enabled: true,
            variance_threshold: 1e-3,
            collect_timing: false,
        };
        assert!(validate_desc(&desc).is_err());
    }

    #[test]
    fn atmosphere_parameter_domain_matches_canonical_config() {
        let base = AetherSpectralReferenceDesc {
            heights: vec![0.0; 4],
            dem_width: 2,
            dem_height: 2,
            spacing: (1.0, 1.0),
            exaggeration: 1.0,
            cam_origin: [0.0, 1.0, 0.0],
            cam_look_at: [1.0, 1.0, 0.0],
            cam_up: [0.0, 1.0, 0.0],
            fov_y_deg: 45.0,
            sun_azimuth_deg: 90.0,
            sun_elevation_deg: 10.0,
            sun_intensity: 1.0,
            turbidity: 2.0,
            ozone_du: 0.0,
            mie_g: 0.0,
            ground_albedo: 0.3,
            width: 1,
            height: 1,
            seed: 7,
            spp: 1,
            enabled: true,
            variance_threshold: 1e-3,
            collect_timing: false,
        };
        assert!(validate_desc(&base).is_ok());
        let mut upper = base.clone();
        upper.ozone_du = 600.0;
        upper.mie_g = 0.99;
        assert!(validate_desc(&upper).is_ok());
        let mut negative_mie = base.clone();
        negative_mie.mie_g = -0.001;
        assert!(validate_desc(&negative_mie).is_err());
        let mut excess_ozone = base;
        excess_ozone.ozone_du = 600.001;
        assert!(validate_desc(&excess_ozone).is_err());
    }
}
