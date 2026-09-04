use super::*;
use crate::core::atmosphere::AETHER_RADIOMETRIC_SCALE_MAX;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedTexture,
};
use crate::terrain::hosek_sky::hosek_rgb_sky;

pub(super) mod luts;

use luts::{AtmosphereGpuLuts, AETHER_LUT_CACHE_CAPACITY};

pub(super) struct AtmosphereInitResources {
    pub(super) sky_bind_group_layout0: wgpu::BindGroupLayout,
    pub(super) sky_bind_group_layout1: wgpu::BindGroupLayout,
    pub(super) sky_pipeline: wgpu::ComputePipeline,
    pub(super) aether_sky_bind_group_layout2: wgpu::BindGroupLayout,
    pub(super) aether_sky_pipeline: wgpu::ComputePipeline,
    pub(super) atmosphere_lut_cache: Mutex<Vec<AtmosphereGpuLuts>>,
    pub(super) sky_fallback_texture: TrackedTexture,
    pub(super) sky_fallback_view: wgpu::TextureView,
    pub(super) scattering_fallback_texture: TrackedTexture,
    pub(super) scattering_fallback_view: wgpu::TextureView,
}

pub(super) struct RenderedSky {
    pub(super) texture: TrackedTexture,
    pub(super) view: wgpu::TextureView,
    pub(super) scattering_view: Option<wgpu::TextureView>,
    /// True only for AETHER. Legacy analytic skies retain their historical
    /// display-referred resolve even though they share the rgba16float target.
    pub(super) linear_hdr: bool,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainSkyUniforms {
    sun_direction_turbidity: [f32; 4],
    ground_albedo_sun_size_sun_intensity_exposure: [f32; 4],
    model_pad: [u32; 4],
    hosek_coeffs_a_d: [[f32; 4]; 3],
    hosek_coeffs_e_h: [[f32; 4]; 3],
    hosek_coeff_i: [f32; 4],
    hosek_radiance: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainSkyCameraUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    eye_position: [f32; 3],
    _pad0: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AtmosphereScatteringUniforms {
    planet_radii_path: [f32; 4],
    mie_ground_scales: [f32; 4],
    sun_direction_intensity: [f32; 4],
    camera_exposure_density_model: [f32; 4],
    lut_dimensions0: [u32; 4],
    lut_dimensions1: [u32; 4],
}

pub(super) fn create_atmosphere_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<AtmosphereInitResources> {
    let sky_bind_group_layout0 =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.sky.bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

    let sky_bind_group_layout1 =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.sky.bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let sky_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "terrain.sky.shader",
        include_str!("../../shaders/sky.wgsl"),
    );

    let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain.sky.pipeline_layout"),
        bind_group_layouts: &[&sky_bind_group_layout0, &sky_bind_group_layout1],
        push_constant_ranges: &[],
    });

    let sky_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("terrain.sky.pipeline"),
            layout: Some(&sky_pipeline_layout),
            module: &sky_shader,
            entry_point: "cs_render_sky",
        },
    )
    .map_err(|message| anyhow!("terrain.sky.pipeline: {message}"))?;

    let aether_sky_bind_group_layout2 =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.aether.sky.bgl2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
    let aether_source = crate::shader_sources::aether_sky();
    let aether_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "terrain.aether.sky.shader",
        &aether_source,
    );
    let aether_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain.aether.sky.pipeline_layout"),
        bind_group_layouts: &[
            &sky_bind_group_layout0,
            &sky_bind_group_layout1,
            &aether_sky_bind_group_layout2,
        ],
        push_constant_ranges: &[],
    });
    let aether_sky_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("terrain.aether.sky.pipeline"),
            layout: Some(&aether_pipeline_layout),
            module: &aether_shader,
            entry_point: "cs_render_aether_sky",
        },
    )
    .map_err(|error| anyhow!("AETHER sky pipeline validation failed: {error}"))?;

    let sky_fallback_texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("terrain.sky.fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
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
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &sky_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[26, 26, 38, 255],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let sky_fallback_view =
        sky_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let scattering_fallback_texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("terrain.aether.scattering.fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )?;
    let scattering_fallback_view =
        scattering_fallback_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.aether.scattering.fallback-view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

    Ok(AtmosphereInitResources {
        sky_bind_group_layout0,
        sky_bind_group_layout1,
        sky_pipeline,
        aether_sky_bind_group_layout2,
        aether_sky_pipeline,
        atmosphere_lut_cache: Mutex::new(Vec::new()),
        sky_fallback_texture,
        sky_fallback_view,
        scattering_fallback_texture,
        scattering_fallback_view,
    })
}

impl TerrainScene {
    pub(super) fn render_sky_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        eye: glam::Vec3,
        width: u32,
        height: u32,
    ) -> Result<Option<RenderedSky>> {
        if !decoded.sky.enabled || width == 0 || height == 0 {
            return Ok(None);
        }
        let aether_sun_intensity = decoded
            .sky
            .sun_intensity
            .clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let aether_exposure = decoded
            .sky
            .sky_exposure
            .clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let sky_sun_y = decoded.light.direction[2].clamp(0.0, 1.0);
        let solar_elevation = sky_sun_y.asin().clamp(0.0, std::f32::consts::FRAC_PI_2);
        let hosek = hosek_rgb_sky(
            decoded.sky.turbidity.clamp(1.0, 10.0),
            decoded.sky.ground_albedo.clamp(0.0, 1.0),
            solar_elevation,
        );
        let sky_uniforms = TerrainSkyUniforms {
            // sky.wgsl is authored in a Y-up frame while terrain lighting is Z-up.
            // Swizzle the decoded terrain light so the sky disk still tracks the
            // terrain sun direction on screen.
            sun_direction_turbidity: [
                decoded.light.direction[0],
                decoded.light.direction[2],
                decoded.light.direction[1],
                decoded.sky.turbidity.clamp(1.0, 10.0),
            ],
            ground_albedo_sun_size_sun_intensity_exposure: [
                decoded.sky.ground_albedo.clamp(0.0, 1.0),
                decoded.sky.sun_size.max(0.0),
                aether_sun_intensity,
                aether_exposure,
            ],
            model_pad: [decoded.sky.model, 0, 0, 0],
            hosek_coeffs_a_d: hosek.uniform_a_d(),
            hosek_coeffs_e_h: hosek.uniform_e_h(),
            hosek_coeff_i: hosek.uniform_i(),
            hosek_radiance: hosek.uniform_radiance(),
        };
        let sky_params = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.sky.params"),
                contents: bytemuck::bytes_of(&sky_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let sky_camera_uniforms = TerrainSkyCameraUniforms {
            view: view_matrix.to_cols_array_2d(),
            proj: proj_matrix.to_cols_array_2d(),
            inv_view: view_matrix.inverse().to_cols_array_2d(),
            inv_proj: proj_matrix.inverse().to_cols_array_2d(),
            eye_position: eye.to_array(),
            _pad0: 0.0,
        };
        let sky_camera = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.sky.camera"),
                contents: bytemuck::bytes_of(&sky_camera_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let sky_texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.sky.output"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let sky_view = sky_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sky_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.sky.bg0"),
            layout: &self.sky_bind_group_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sky_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
            ],
        });
        let sky_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.sky.bg1"),
            layout: &self.sky_bind_group_layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_camera.as_entire_binding(),
            }],
        });

        let gx = (width + 7) / 8;
        let gy = (height + 7) / 8;
        let mut material_scattering_view = None;
        if decoded.sky.model == 3 {
            let lut_handle = decoded.sky.lut_handle.as_ref().ok_or_else(|| {
                anyhow!(
                    "AETHER TerrainRenderer has no typed LUT handle; refusing to render with an implicit or legacy atmosphere"
                )
            })?;
            let config = lut_handle.config();
            let mut cache = self
                .atmosphere_lut_cache
                .lock()
                .map_err(|_| anyhow!("AETHER LUT cache mutex poisoned"))?;
            let key = lut_handle.deterministic_sha256();
            let cache_index = match cache
                .iter()
                .position(|entry| entry.deterministic_sha256 == key)
            {
                Some(index) => index,
                None => {
                    if cache.len() >= AETHER_LUT_CACHE_CAPACITY {
                        return Err(anyhow!(
                            "AETHER LUT cache capacity ({AETHER_LUT_CACHE_CAPACITY}) exhausted; refusing an untracked fallback"
                        ));
                    }
                    let uploaded = AtmosphereGpuLuts::upload(
                        self.device.as_ref(),
                        self.queue.as_ref(),
                        lut_handle,
                    )?;
                    log::debug!(
                        "uploaded {} bytes of tracked AETHER LUT textures for {} payload {}",
                        uploaded.byte_size(),
                        if lut_handle.luts().metadata.precomputed {
                            "shipped"
                        } else {
                            "custom-baked"
                        },
                        lut_handle.deterministic_sha256_hex(),
                    );
                    cache.push(uploaded);
                    cache.len() - 1
                }
            };
            let gpu_luts = &cache[cache_index];
            material_scattering_view = Some(gpu_luts.scattering_view());
            let dimensions = gpu_luts.dimensions;
            let atmosphere_uniforms = AtmosphereScatteringUniforms {
                planet_radii_path: [
                    config.bottom_radius_m,
                    config.top_radius_m,
                    config.max_aerial_distance_m,
                    config.ozone_du,
                ],
                mie_ground_scales: [
                    config.mie_g,
                    config.ground_albedo,
                    config.rayleigh_scale_height_m,
                    config.mie_scale_height_m,
                ],
                sun_direction_intensity: [
                    decoded.light.direction[0],
                    decoded.light.direction[2],
                    decoded.light.direction[1],
                    aether_sun_intensity,
                ],
                // Terrain world space is Z-up; the sky module's radiometric
                // lookup is authored Y-up, hence eye.z supplies altitude.
                camera_exposure_density_model: [
                    eye.z.max(0.0),
                    aether_exposure,
                    decoded.sky.aerial_density,
                    3.0,
                ],
                lut_dimensions0: [
                    dimensions.transmittance_mu,
                    dimensions.transmittance_height,
                    dimensions.scattering_mu_view,
                    dimensions.scattering_mu_sun,
                ],
                lut_dimensions1: [
                    dimensions.scattering_height,
                    dimensions.scattering_nu,
                    dimensions.aerial_distance,
                    dimensions.aerial_mu_view,
                ],
            };
            let atmosphere_buffer = tracked_create_buffer_init(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("terrain.aether.sky.uniforms"),
                    contents: bytemuck::bytes_of(&atmosphere_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            )?;
            let aether_bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.aether.sky.bg2"),
                layout: &self.aether_sky_bind_group_layout2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atmosphere_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gpu_luts.transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gpu_luts.scattering_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&gpu_luts.aerial_view),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.aether.sky.compute"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("terrain.aether.sky.shader");
            cpass.set_pipeline(&self.aether_sky_pipeline);
            cpass.set_bind_group(0, &sky_bg0, &[]);
            cpass.set_bind_group(1, &sky_bg1, &[]);
            cpass.set_bind_group(2, &aether_bg2, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        } else {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.sky.compute"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("terrain.sky.shader");
            cpass.set_pipeline(&self.sky_pipeline);
            cpass.set_bind_group(0, &sky_bg0, &[]);
            cpass.set_bind_group(1, &sky_bg1, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        Ok(Some(RenderedSky {
            texture: sky_texture,
            view: sky_view,
            scattering_view: material_scattering_view,
            linear_hdr: decoded.sky.model == 3,
        }))
    }
}
