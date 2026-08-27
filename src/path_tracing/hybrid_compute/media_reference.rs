use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::core::gpu_timing::OneShotTiming;
use crate::core::resource_tracker::TrackedBuffer;
use crate::media::{
    ratio_track, trace_reference_sample, Bounds3, DensityField, DirectionalSun,
    EnvironmentDistribution, MediaError, Ray, ReferenceMediumInterval, ReferenceScene,
    ReferenceSurfaceHit, ReferenceTransportConfig, Rgb, SampleIdentity, TrackingContext,
};

use super::terrain_heightfield::{EarthCurvatureUniforms, TerrainPtScene};
use super::{
    tracked_create_buffer, tracked_create_buffer_init, try_ctx, HybridPathTracer,
    TerrainReferenceDesc,
};

const QUERY_SHADER: &str = include_str!("../../shaders/nephele_terrain_trace_adapter.wgsl");
const TERRAIN_TRACE_SHADER_LABEL: &str = "nephele.terrain_trace.adapter";
const TERRAIN_QUERY_REACH: f32 = 1.0e30;
const REALTIME_FROXEL_DEPTH_SLICES: f32 = 64.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRay {
    origin_tmin: [f32; 4],
    direction_tmax: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHit {
    point_t: [f32; 4],
    normal_hit: [f32; 4],
}

enum MediumDomain {
    Homogeneous { maximum_distance: f32 },
    Bounded(Bounds3),
}

pub struct TerrainMediaReferenceOutput {
    pub beauty: Vec<f32>,
    pub transmittance: Vec<f32>,
    pub in_scatter: Vec<f32>,
    pub cloud_shadow: Vec<f32>,
    pub optical_depth: Vec<f32>,
    pub terrain_slice: Vec<f32>,
    /// Exact primary-ray terrain classification: 0 = miss, 1 = hit.
    pub terrain_hit: Vec<u8>,
    /// Reference-only sun visibility at the camera-medium interval midpoint:
    /// 0 = no positive interval, 1 = terrain-blocked, 2 = terrain-visible.
    pub media_lighting_visibility: Vec<u8>,
    pub sample_count: u64,
    pub step_count: u64,
    pub majorant_proof: crate::media::MajorantProof,
    pub executed_multi_scatter: bool,
    pub host_visible_bytes: u64,
    pub adapter: String,
    pub backend: String,
    pub driver: String,
}

impl MediumDomain {
    fn new(context: &TrackingContext, homogeneous_reach: f32) -> Result<Self, MediaError> {
        match context.medium().density() {
            DensityField::Homogeneous(_) => {
                if !homogeneous_reach.is_finite() || homogeneous_reach <= 0.0 {
                    return Err(MediaError::InvalidTransport(
                        "homogeneous reference medium reach must be finite and positive".into(),
                    ));
                }
                Ok(Self::Homogeneous {
                    maximum_distance: homogeneous_reach,
                })
            }
            DensityField::PerlinWorley(field) => Ok(Self::Bounded(field.transform.bounds)),
            DensityField::Grid3D(field) => Ok(Self::Bounded(field.transform().bounds)),
        }
    }

    fn interval(&self, ray: Ray, maximum_distance: f32) -> Option<ReferenceMediumInterval> {
        match self {
            Self::Homogeneous {
                maximum_distance: reach,
            } => Some(ReferenceMediumInterval {
                start: 0.0,
                end: maximum_distance.min(*reach),
            }),
            Self::Bounded(bounds) => ray_box_interval(ray, *bounds, maximum_distance),
        }
    }
}

/// Slow production adapter used by the integrated reference. Each geometry
/// query dispatches an entry that calls the exact assembled PROMETHEUS
/// `terrain_trace`; no CPU terrain marcher is substituted.
struct TerrainTraceReferenceScene {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    empty0: wgpu::BindGroup,
    empty1: wgpu::BindGroup,
    terrain_group: wgpu::BindGroup,
    query_group: wgpu::BindGroup,
    ray_buffer: TrackedBuffer,
    hit_buffer: TrackedBuffer,
    readback: TrackedBuffer,
    _terrain_buffer: TrackedBuffer,
    _curvature_buffer: TrackedBuffer,
    _terrain_scene: TerrainPtScene,
    albedo: Rgb,
    medium_domain: MediumDomain,
    timing: RefCell<Option<OneShotTiming>>,
    timing_recorded: Cell<bool>,
}

impl TerrainTraceReferenceScene {
    fn new(
        desc: &TerrainReferenceDesc,
        context: &TrackingContext,
        homogeneous_reach: f32,
    ) -> Result<Self, MediaError> {
        if desc.mesh.is_some() {
            return Err(MediaError::InvalidTransport(
                "the terrain_trace media adapter accepts terrain-only reference scenes".into(),
            ));
        }
        let gpu = try_ctx().map_err(render_error)?;
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let metal_backend = gpu.adapter.get_info().backend == wgpu::Backend::Metal;
        let terrain_scene = TerrainPtScene::new(
            &device,
            &queue,
            &desc.heights,
            desc.dem_width,
            desc.dem_height,
            desc.spacing,
            desc.exaggeration,
            desc.albedo,
            desc.env_map
                .as_ref()
                .map(|(data, width, height)| (data.as_slice(), *width, *height)),
            desc.env_intensity,
        )
        .map_err(render_error)?;
        let terrain_uniform = terrain_scene.uniforms(1, 2);
        let curvature_uniform = EarthCurvatureUniforms::new(
            desc.earth_model,
            desc.refraction_model,
            desc.observer_geodetic_deg,
            f64::from(desc.sun_azimuth_deg),
        )
        .map_err(render_error)?;
        let terrain_buffer = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("nephele-terrain-trace-uniform"),
                contents: bytemuck::bytes_of(&terrain_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )
        .map_err(render_error)?;
        let curvature_buffer = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("nephele-terrain-trace-curvature"),
                contents: bytemuck::bytes_of(&curvature_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )
        .map_err(render_error)?;
        let ray_buffer = tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("nephele-terrain-trace-ray"),
                size: std::mem::size_of::<GpuRay>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .map_err(render_error)?;
        let hit_buffer = tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("nephele-terrain-trace-hit"),
                size: std::mem::size_of::<GpuHit>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .map_err(render_error)?;
        let readback = tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("nephele-terrain-trace-readback"),
                size: std::mem::size_of::<GpuHit>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .map_err(render_error)?;

        let source = format!("{}\n{QUERY_SHADER}", crate::shader_sources::hybrid_kernel());
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            TERRAIN_TRACE_SHADER_LABEL,
            &source,
        );
        let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            &device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("nephele-terrain-trace-adapter"),
                layout: None,
                module: &shader,
                entry_point: "main_nephele_terrain_trace_adapter",
            },
        )
        .map_err(|error| {
            MediaError::InvalidTransport(format!(
                "terrain_trace adapter pipeline validation failed: {error}"
            ))
        })?;
        let empty0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele-terrain-trace-empty0"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[],
        });
        let empty1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele-terrain-trace-empty1"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[],
        });
        let height_view = terrain_scene
            .pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = terrain_scene
            .pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // wgpu 0.19's Metal auto-layout assigns these adjacent sampled-texture
        // slots in reverse for the assembled adapter entry. Keep the canonical
        // WGSL bindings on other backends and compensate only for that measured
        // backend behavior here.
        let (height_binding, minmax_binding) = if metal_backend {
            (&minmax_view, &height_view)
        } else {
            (&height_view, &minmax_view)
        };
        let terrain_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele-terrain-trace-terrain"),
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_binding),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(minmax_binding),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: curvature_buffer.as_entire_binding(),
                },
            ],
        });
        let query_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele-terrain-trace-query"),
            layout: &pipeline.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: ray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: hit_buffer.as_entire_binding(),
                },
            ],
        });

        let timing = OneShotTiming::for_device(device.clone(), queue.clone());
        Ok(Self {
            device,
            queue,
            pipeline,
            empty0,
            empty1,
            terrain_group,
            query_group,
            ray_buffer,
            hit_buffer,
            readback,
            _terrain_buffer: terrain_buffer,
            _curvature_buffer: curvature_buffer,
            _terrain_scene: terrain_scene,
            albedo: Rgb::new(desc.albedo, "terrain albedo")?,
            medium_domain: MediumDomain::new(context, homogeneous_reach)?,
            timing: RefCell::new(Some(timing)),
            timing_recorded: Cell::new(false),
        })
    }

    fn query(&self, ray: Ray, maximum_distance: f32) -> Result<Option<GpuHit>, MediaError> {
        if !maximum_distance.is_finite() || maximum_distance < 0.0 {
            return Err(MediaError::InvalidTransport(
                "terrain query reach must be finite and non-negative".into(),
            ));
        }
        let input = GpuRay {
            origin_tmin: [ray.origin[0], ray.origin[1], ray.origin[2], 1.0e-3],
            direction_tmax: [
                ray.direction[0],
                ray.direction[1],
                ray.direction[2],
                maximum_distance,
            ],
        };
        self.queue
            .write_buffer(&self.ray_buffer, 0, bytemuck::bytes_of(&input));
        crate::core::shader_registry::record_shader_use(TERRAIN_TRACE_SHADER_LABEL);
        let mut timing = if self.timing_recorded.get() {
            None
        } else {
            self.timing.borrow_mut().take()
        };
        let timing_live = timing.as_ref().is_some_and(OneShotTiming::is_live);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nephele-terrain-trace-query"),
            });
        let timing_scope = timing
            .as_mut()
            .and_then(|timing| timing.begin(&mut encoder, TERRAIN_TRACE_SHADER_LABEL));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nephele-terrain-trace-query"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.empty0, &[]);
        pass.set_bind_group(1, &self.empty1, &[]);
        pass.set_bind_group(2, &self.terrain_group, &[]);
        pass.set_bind_group(3, &self.query_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        if let Some(timing) = timing.as_mut() {
            timing.end(&mut encoder, timing_scope, 1);
            timing.resolve(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::Maintain::Wait);
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nephele-terrain-trace-readback-copy"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.hit_buffer,
            0,
            &self.readback,
            0,
            std::mem::size_of::<GpuHit>() as u64,
        );
        self.queue.submit([copy_encoder.finish()]);
        let slice = self.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| MediaError::InvalidTransport(error.to_string()))?
            .map_err(|error| MediaError::InvalidTransport(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let hit = *bytemuck::from_bytes::<GpuHit>(&mapped);
        drop(mapped);
        self.readback.unmap();
        if let Some(timing) = timing {
            self.timing_recorded.set(true);
            let recorded = timing_live && timing.record_into_certificate();
            if !recorded {
                if !timing_live {
                    crate::core::degradation::record_degradation(
                        "timing_unavailable",
                        TERRAIN_TRACE_SHADER_LABEL,
                        "timestamp query unavailable or failed for the media terrain-trace pass; certificate passes[].gpu_ms reported as 0",
                    );
                }
                crate::core::certificate::record_pass(TERRAIN_TRACE_SHADER_LABEL, 0.0, 1);
            }
        }
        Ok((hit.normal_hit[3] > 0.5).then_some(hit))
    }
}

impl ReferenceScene for TerrainTraceReferenceScene {
    fn intersect(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceSurfaceHit>, MediaError> {
        self.query(ray, maximum_distance)?.map_or(Ok(None), |hit| {
            Ok(Some(ReferenceSurfaceHit {
                distance: hit.point_t[3],
                position: [hit.point_t[0], hit.point_t[1], hit.point_t[2]],
                normal: [hit.normal_hit[0], hit.normal_hit[1], hit.normal_hit[2]],
                albedo: self.albedo,
            }))
        })
    }

    fn occluded(&self, ray: Ray, maximum_distance: f32) -> Result<bool, MediaError> {
        Ok(self.query(ray, maximum_distance)?.is_some())
    }

    fn geometry_reach(&self, _ray: Ray) -> Result<f32, MediaError> {
        Ok(TERRAIN_QUERY_REACH)
    }

    fn medium_interval(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceMediumInterval>, MediaError> {
        Ok(self.medium_domain.interval(ray, maximum_distance))
    }
}

impl HybridPathTracer {
    /// Render the integrated terrain/media reference for every acceptance
    /// pixel and sample. The production `terrain_trace` adapter and canonical
    /// stochastic transport are both exercised by every beauty sample.
    pub fn render_terrain_media_reference(
        &self,
        terrain: &TerrainReferenceDesc,
        context: &TrackingContext,
        homogeneous_medium_reach: f32,
        environment: &EnvironmentDistribution,
        sun: DirectionalSun,
        samples_per_pixel: u64,
        froxel_depth_range: (f32, f32),
        config: ReferenceTransportConfig,
        full_viewport: [u32; 2],
        crop_origin: [u32; 2],
        camera_basis: [[f32; 3]; 3],
    ) -> Result<TerrainMediaReferenceOutput, MediaError> {
        if terrain.width == 0 || terrain.height == 0 || samples_per_pixel == 0 {
            return Err(MediaError::InvalidTransport(
                "reference width, height, and samples_per_pixel must be positive".into(),
            ));
        }
        validate_reference_crop([terrain.width, terrain.height], full_viewport, crop_origin)?;
        let pixel_count = usize::try_from(
            u64::from(terrain.width)
                .checked_mul(u64::from(terrain.height))
                .ok_or_else(|| {
                    MediaError::InvalidTransport("reference pixel count overflowed".into())
                })?,
        )
        .map_err(|_| MediaError::InvalidTransport("reference pixel count overflowed".into()))?;
        let scene = TerrainTraceReferenceScene::new(terrain, context, homogeneous_medium_reach)?;
        let [forward, right, up] = camera_basis.map(glam::Vec3::from);
        if [forward, right, up]
            .iter()
            .any(|axis| !axis.is_finite() || *axis == glam::Vec3::ZERO)
        {
            return Err(MediaError::InvalidTransport(
                "reference camera basis is degenerate".into(),
            ));
        }
        let tan_half_fov = (terrain.fov_y_deg.to_radians() * 0.5).tan();
        if !tan_half_fov.is_finite() || tan_half_fov <= 0.0 {
            return Err(MediaError::InvalidTransport(
                "reference fov_y_deg must be finite and in (0, 180)".into(),
            ));
        }
        let aspect = full_viewport[0] as f32 / full_viewport[1] as f32;
        let mut beauty = vec![0.0; pixel_count * 3];
        let mut transmittance = vec![0.0; pixel_count * 3];
        let mut in_scatter = vec![0.0; pixel_count * 3];
        let mut cloud_shadow = vec![0.0; pixel_count * 3];
        let mut terrain_slice = vec![0.0; pixel_count];
        let mut terrain_hit = vec![0; pixel_count];
        let mut media_lighting_visibility = vec![0; pixel_count];
        let mut step_count = 0u64;
        let mut executed_multi_scatter = false;
        for y in 0..terrain.height {
            for x in 0..terrain.width {
                let pixel = u64::from(y) * u64::from(terrain.width) + u64::from(x);
                let absolute_x = crop_origin[0] + x;
                let absolute_y = crop_origin[1] + y;
                let sample_pixel =
                    u64::from(absolute_y) * u64::from(full_viewport[0]) + u64::from(absolute_x);
                let ndc =
                    reference_pixel_ndc(full_viewport, crop_origin, [x, y], aspect, tan_half_fov);
                let ray = Ray {
                    origin: terrain.cam_origin,
                    direction: (forward + right * ndc.x + up * ndc.y)
                        .normalize()
                        .to_array(),
                };
                let maximum = scene.geometry_reach(ray)?;
                let surface = scene.intersect(ray, maximum)?;
                let distance = surface.map_or(maximum, |hit| hit.distance);
                terrain_hit[pixel as usize] = u8::from(surface.is_some());
                terrain_slice[pixel as usize] = if surface.is_some() {
                    froxel_slice_coordinate(distance, froxel_depth_range.0, froxel_depth_range.1)?
                } else {
                    REALTIME_FROXEL_DEPTH_SLICES
                };
                media_lighting_visibility[pixel as usize] =
                    classify_media_lighting(&scene, ray, distance, sun.direction_to_sun)?;
                for sample in 0..samples_per_pixel {
                    let identity = SampleIdentity {
                        frame: u64::from(terrain.seed),
                        pixel: sample_pixel,
                        sample,
                        bounce: 0,
                    };
                    let transport = trace_reference_sample(
                        context,
                        &scene,
                        environment,
                        sun,
                        ray,
                        identity,
                        config,
                    )?;
                    step_count = step_count.saturating_add(transport.tracking_step_count);
                    executed_multi_scatter |= transport.collision_count > 1;
                    let (camera_t, steps) = tracked_transmittance(
                        context,
                        &scene,
                        ray,
                        distance,
                        SampleIdentity {
                            sample: sample.wrapping_add(0x4000),
                            ..identity
                        },
                    )?;
                    step_count = step_count.saturating_add(steps);
                    let (sun_t, steps) = if let Some(hit) = surface {
                        let sun_ray = Ray {
                            origin: hit.position,
                            direction: sun.direction_to_sun,
                        };
                        tracked_transmittance(
                            context,
                            &scene,
                            sun_ray,
                            scene.geometry_reach(sun_ray)?,
                            SampleIdentity {
                                sample: sample.wrapping_add(0x5000),
                                ..identity
                            },
                        )?
                    } else {
                        (Rgb::ONE, 0)
                    };
                    step_count = step_count.saturating_add(steps);
                    let index = pixel as usize * 3;
                    for channel in 0..3 {
                        beauty[index + channel] += transport.radiance.components()[channel];
                        transmittance[index + channel] += camera_t.components()[channel];
                        cloud_shadow[index + channel] += sun_t.components()[channel];
                        if transport.collision_count > 0 {
                            in_scatter[index + channel] += transport.radiance.components()[channel];
                        }
                    }
                }
            }
        }
        let scale = 1.0 / samples_per_pixel as f32;
        for values in [
            &mut beauty,
            &mut transmittance,
            &mut in_scatter,
            &mut cloud_shadow,
        ] {
            for value in values {
                *value *= scale;
            }
        }
        let optical_depth = transmittance
            .iter()
            .map(|value| -value.max(f32::MIN_POSITIVE).ln())
            .collect();
        let info = try_ctx().map_err(render_error)?.adapter.get_info();
        Ok(TerrainMediaReferenceOutput {
            beauty,
            transmittance,
            in_scatter,
            cloud_shadow,
            optical_depth,
            terrain_slice,
            terrain_hit,
            media_lighting_visibility,
            sample_count: pixel_count as u64 * samples_per_pixel,
            step_count,
            majorant_proof: context.majorant().proof().clone(),
            executed_multi_scatter,
            host_visible_bytes: pixel_count as u64 * (5 * 3 * 4 + 4 + 2),
            adapter: info.name,
            backend: format!("{:?}", info.backend),
            driver: [info.driver, info.driver_info]
                .into_iter()
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        })
    }
}

fn classify_media_lighting<S: ReferenceScene>(
    scene: &S,
    camera_ray: Ray,
    maximum_distance: f32,
    direction_to_sun: [f32; 3],
) -> Result<u8, MediaError> {
    let Some(interval) = scene.medium_interval(camera_ray, maximum_distance)? else {
        return Ok(0);
    };
    if interval.end <= interval.start {
        return Ok(0);
    }
    let midpoint = interval.start + 0.5 * (interval.end - interval.start);
    let sun_ray = Ray {
        origin: [
            camera_ray.origin[0] + camera_ray.direction[0] * midpoint,
            camera_ray.origin[1] + camera_ray.direction[1] * midpoint,
            camera_ray.origin[2] + camera_ray.direction[2] * midpoint,
        ],
        direction: direction_to_sun,
    };
    Ok(
        if scene.occluded(sun_ray, scene.geometry_reach(sun_ray)?)? {
            1
        } else {
            2
        },
    )
}

fn validate_reference_crop(
    output_size: [u32; 2],
    full_viewport: [u32; 2],
    crop_origin: [u32; 2],
) -> Result<(), MediaError> {
    if full_viewport.contains(&0) {
        return Err(MediaError::InvalidTransport(
            "reference full viewport dimensions must be positive".into(),
        ));
    }
    let crop_end = [
        crop_origin[0].checked_add(output_size[0]),
        crop_origin[1].checked_add(output_size[1]),
    ];
    if crop_end[0].is_none_or(|end| end > full_viewport[0])
        || crop_end[1].is_none_or(|end| end > full_viewport[1])
    {
        return Err(MediaError::InvalidTransport(
            "reference crop must lie within the full viewport".into(),
        ));
    }
    Ok(())
}

fn reference_pixel_ndc(
    full_viewport: [u32; 2],
    crop_origin: [u32; 2],
    output_pixel: [u32; 2],
    aspect: f32,
    tan_half_fov: f32,
) -> glam::Vec2 {
    let x = crop_origin[0] + output_pixel[0];
    let y = crop_origin[1] + output_pixel[1];
    glam::Vec2::new(
        (2.0 * (x as f32 + 0.5) / full_viewport[0] as f32 - 1.0) * aspect * tan_half_fov,
        (1.0 - 2.0 * (y as f32 + 0.5) / full_viewport[1] as f32) * tan_half_fov,
    )
}

fn froxel_slice_coordinate(distance: f32, near: f32, far: f32) -> Result<f32, MediaError> {
    if !distance.is_finite()
        || distance < 0.0
        || !near.is_finite()
        || !far.is_finite()
        || near <= 0.0
        || far <= near
    {
        return Err(MediaError::InvalidTransport(
            "froxel slice mapping requires finite distance >= 0 and 0 < near < far".into(),
        ));
    }
    let unit = ((distance.max(near) / near).ln() / (far / near).ln()).clamp(0.0, 1.0);
    Ok((unit * REALTIME_FROXEL_DEPTH_SLICES).min(REALTIME_FROXEL_DEPTH_SLICES - 1.0))
}

#[cfg(test)]
mod froxel_slice_tests {
    use super::*;

    struct LightingScene {
        interval: Option<ReferenceMediumInterval>,
        blocked: bool,
    }

    impl ReferenceScene for LightingScene {
        fn intersect(
            &self,
            _ray: Ray,
            _maximum_distance: f32,
        ) -> Result<Option<ReferenceSurfaceHit>, MediaError> {
            Ok(None)
        }

        fn occluded(&self, _ray: Ray, _maximum_distance: f32) -> Result<bool, MediaError> {
            Ok(self.blocked)
        }

        fn geometry_reach(&self, _ray: Ray) -> Result<f32, MediaError> {
            Ok(100.0)
        }

        fn medium_interval(
            &self,
            _ray: Ray,
            _maximum_distance: f32,
        ) -> Result<Option<ReferenceMediumInterval>, MediaError> {
            Ok(self.interval)
        }
    }

    #[test]
    fn reference_terrain_distance_uses_realtime_logarithmic_slice_mapping() {
        let near = 0.1;
        let far = 1000.0;
        assert_eq!(froxel_slice_coordinate(near, near, far).unwrap(), 0.0);
        assert_eq!(froxel_slice_coordinate(far, near, far).unwrap(), 63.0);
        let geometric_midpoint = (near * far).sqrt();
        assert_eq!(
            froxel_slice_coordinate(geometric_midpoint, near, far).unwrap(),
            32.0
        );
    }

    #[test]
    fn crop_ray_matches_corresponding_full_frame_ray() {
        let aspect = 1.0;
        let tan_half_fov = (45.0f32.to_radians() * 0.5).tan();
        let full = reference_pixel_ndc([64, 64], [0, 0], [40, 45], aspect, tan_half_fov);
        let crop = reference_pixel_ndc([64, 64], [24, 24], [16, 21], aspect, tan_half_fov);
        let forward = glam::Vec3::NEG_Z;
        let right = glam::Vec3::X;
        let up = glam::Vec3::Y;
        assert_eq!(
            (forward + right * full.x + up * full.y).normalize(),
            (forward + right * crop.x + up * crop.y).normalize()
        );
    }

    #[test]
    fn invalid_reference_crop_bounds_fail_closed() {
        assert!(validate_reference_crop([16, 16], [64, 64], [48, 48]).is_ok());
        assert!(validate_reference_crop([16, 16], [64, 64], [49, 48]).is_err());
        assert!(validate_reference_crop([16, 16], [0, 64], [0, 0]).is_err());
        assert!(validate_reference_crop([16, 16], [64, 64], [u32::MAX, 0]).is_err());
    }

    #[test]
    fn media_lighting_classification_is_exact_and_reference_only() {
        let ray = Ray {
            origin: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
        };
        let no_medium = LightingScene {
            interval: None,
            blocked: true,
        };
        assert_eq!(
            classify_media_lighting(&no_medium, ray, 10.0, [0.0, 1.0, 0.0]).unwrap(),
            0
        );
        for (blocked, expected) in [(true, 1), (false, 2)] {
            let scene = LightingScene {
                interval: Some(ReferenceMediumInterval {
                    start: 2.0,
                    end: 6.0,
                }),
                blocked,
            };
            assert_eq!(
                classify_media_lighting(&scene, ray, 10.0, [0.0, 1.0, 0.0]).unwrap(),
                expected
            );
        }
    }
}

fn tracked_transmittance<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    ray: Ray,
    maximum_distance: f32,
    identity: SampleIdentity,
) -> Result<(Rgb, u64), MediaError> {
    let Some(interval) = scene.medium_interval(ray, maximum_distance)? else {
        return Ok((Rgb::ONE, 0));
    };
    ratio_track(
        context,
        Ray {
            origin: [
                ray.origin[0] + ray.direction[0] * interval.start,
                ray.origin[1] + ray.direction[1] * interval.start,
                ray.origin[2] + ray.direction[2] * interval.start,
            ],
            direction: ray.direction,
        },
        interval.end - interval.start,
        identity,
    )
}

fn ray_box_interval(
    ray: Ray,
    bounds: Bounds3,
    maximum_distance: f32,
) -> Option<ReferenceMediumInterval> {
    let mut start = 0.0f32;
    let mut end = maximum_distance;
    for axis in 0..3 {
        if ray.direction[axis] == 0.0 {
            if ray.origin[axis] < bounds.min[axis] || ray.origin[axis] > bounds.max[axis] {
                return None;
            }
            continue;
        }
        let inverse = ray.direction[axis].recip();
        let a = (bounds.min[axis] - ray.origin[axis]) * inverse;
        let b = (bounds.max[axis] - ray.origin[axis]) * inverse;
        start = start.max(a.min(b));
        end = end.min(a.max(b));
        if end < start {
            return None;
        }
    }
    (end > start).then_some(ReferenceMediumInterval { start, end })
}

fn render_error(error: impl std::fmt::Display) -> MediaError {
    MediaError::InvalidTransport(format!("terrain_trace adapter failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{
        DensityMapping, Homogeneous, MajorantGrid, Medium, Phase, ReferenceTransportConfig,
    };

    fn flat_terrain_desc() -> TerrainReferenceDesc {
        TerrainReferenceDesc {
            heights: vec![0.0; 4],
            dem_width: 2,
            dem_height: 2,
            spacing: (2.0, 2.0),
            exaggeration: 1.0,
            albedo: [0.5; 3],
            cam_origin: [0.0, 2.0, 0.0],
            cam_look_at: [0.0, 0.0, 0.0],
            cam_up: [0.0, 0.0, -1.0],
            fov_y_deg: 45.0,
            exposure: 1.0,
            sun_azimuth_deg: 0.0,
            sun_elevation_deg: 45.0,
            sun_intensity: 0.0,
            sun_color: [0.0; 3],
            observer_geodetic_deg: [0.0; 2],
            earth_model: crate::geo::refraction::EarthModel::Flat,
            refraction_model: crate::geo::refraction::RefractionModel::None,
            env_map: None,
            env_intensity: 1.0,
            atmosphere: None,
            mesh: None,
            width: 1,
            height: 1,
            seed: 7,
            spp: 1,
            max_frames: 1,
            min_frames: 1,
            variance_threshold: 1.0,
        }
    }

    #[test]
    fn adapter_shader_calls_the_production_terrain_trace() {
        let source = format!("{}\n{QUERY_SHADER}", crate::shader_sources::hybrid_kernel());
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert!(QUERY_SHADER.contains("let hit = terrain_trace(ray, false, true);"));
        assert!(QUERY_SHADER.contains("nephele_terrain_hits[gid.x].point_t"));
    }

    #[test]
    fn bounded_medium_interval_is_distinct_from_geometry_reach() {
        let interval = ray_box_interval(
            Ray {
                origin: [0.0, 0.0, -4.0],
                direction: [0.0, 0.0, 1.0],
            },
            Bounds3 {
                min: [-1.0; 3],
                max: [1.0; 3],
            },
            TERRAIN_QUERY_REACH,
        )
        .unwrap();
        assert_eq!(
            interval,
            ReferenceMediumInterval {
                start: 3.0,
                end: 5.0
            }
        );
        assert!(interval.end < TERRAIN_QUERY_REACH);
    }

    #[test]
    #[ignore = "requires an available GPU adapter; run in the exact-head physical lane"]
    fn production_terrain_adapter_drives_non_vacuum_transport() {
        let capture =
            crate::core::certificate::begin_render_capture("media.volumetric_reference.timing");
        let medium = Medium::new(
            [1.0e6; 3],
            [0.0; 3],
            Phase::Isotropic,
            DensityField::Homogeneous(Homogeneous {
                authored_density: 1.0,
                mapping: DensityMapping {
                    physical_density_per_authored_unit: 1.0,
                },
            }),
        )
        .unwrap();
        let context =
            TrackingContext::new(medium.clone(), MajorantGrid::construct(&medium, 1).unwrap())
                .unwrap();
        let scene = TerrainTraceReferenceScene::new(&flat_terrain_desc(), &context, 2.0).unwrap();
        let camera_ray = Ray {
            origin: [0.0, 2.0, 0.0],
            direction: [0.0, -1.0, 0.0],
        };
        assert!(scene
            .intersect(camera_ray, TERRAIN_QUERY_REACH)
            .unwrap()
            .is_some());
        let sample = trace_reference_sample(
            &context,
            &scene,
            &EnvironmentDistribution::new(1, 1, vec![Rgb::ONE]).unwrap(),
            DirectionalSun::new([0.0, 1.0, 0.0], [0.0; 3]).unwrap(),
            camera_ray,
            SampleIdentity {
                frame: 0,
                pixel: 0,
                sample: 0,
                bounce: 0,
            },
            ReferenceTransportConfig {
                roulette_start_bounce: 0,
                roulette_minimum_probability: 0.0,
            },
        )
        .unwrap();
        assert_eq!(sample.collision_count, 1);
        assert_eq!(sample.surface_count, 0);
        assert_eq!(sample.radiance, Rgb::ZERO);
        capture.finish();
        let report: serde_json::Value = serde_json::from_str(
            &crate::core::certificate::execution_report_json().expect("certificate assembles"),
        )
        .expect("certificate parses");
        let pass = &report["passes"][0];
        assert_eq!(pass["label"], TERRAIN_TRACE_SHADER_LABEL);
        let gpu_ms = pass["gpu_ms"].as_f64().expect("gpu_ms is numeric");
        eprintln!(
            "[media timing] backend={} gpu_ms={gpu_ms}",
            report["adapter"]["backend"]
        );
        if gpu_ms == 0.0 {
            assert!(report["degradations"].as_array().is_some_and(|entries| {
                entries.iter().any(|entry| {
                    entry["kind"] == "timing_unavailable"
                        && entry["name"] == TERRAIN_TRACE_SHADER_LABEL
                })
            }));
        } else {
            assert!(gpu_ms > 0.0, "live GPU timing must be positive");
        }
    }

    #[test]
    #[ignore = "requires an available GPU adapter; run in the exact-head physical lane"]
    fn oblique_crop_ray_hits_continuous_flat_y_up_heightfield() {
        let medium = Medium::new(
            [0.01; 3],
            [0.01; 3],
            Phase::Isotropic,
            DensityField::Homogeneous(Homogeneous {
                authored_density: 1.0,
                mapping: DensityMapping {
                    physical_density_per_authored_unit: 1.0,
                },
            }),
        )
        .unwrap();
        let context =
            TrackingContext::new(medium.clone(), MajorantGrid::construct(&medium, 1).unwrap())
                .unwrap();
        let mut terrain = flat_terrain_desc();
        terrain.heights = vec![10.0; 16 * 16];
        terrain.dem_width = 16;
        terrain.dem_height = 16;
        terrain.spacing = (4.0, 4.0);
        terrain.cam_origin = [0.0, 46.98463, 17.101007];
        terrain.cam_look_at = [0.0; 3];
        terrain.cam_up = [0.0, 1.0, 0.0];
        let scene = TerrainTraceReferenceScene::new(&terrain, &context, 120.0).unwrap();
        let origin = glam::Vec3::from(terrain.cam_origin);
        let forward = (glam::Vec3::from(terrain.cam_look_at) - origin).normalize();
        let right = forward.cross(glam::Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let mut hits = 0;
        for local_y in 0..16 {
            for local_x in 0..16 {
                let ndc = reference_pixel_ndc(
                    [64, 64],
                    [0, 24],
                    [local_x, local_y],
                    1.0,
                    (45.0f32.to_radians() * 0.5).tan(),
                );
                let direction = (forward + right * ndc.x + up * ndc.y).normalize();
                let plane_t = (10.0 - origin.y) / direction.y;
                let plane_hit = origin + direction * plane_t;
                assert!((-30.0..=30.0).contains(&plane_hit.x));
                assert!((-30.0..=30.0).contains(&plane_hit.z));
                hits += usize::from(
                    scene
                        .intersect(
                            Ray {
                                origin: origin.to_array(),
                                direction: direction.to_array(),
                            },
                            TERRAIN_QUERY_REACH,
                        )
                        .unwrap()
                        .is_some(),
                );
            }
        }
        assert_eq!(hits, 16 * 16);
    }
}
