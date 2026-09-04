use super::*;

#[derive(Debug)]
pub(super) struct PbrIblResources {
    pub(super) _irradiance_texture: TrackedTexture,
    pub(super) irradiance_view: TextureView,
    pub(super) irradiance_sampler: Sampler,
    pub(super) _prefilter_texture: TrackedTexture,
    pub(super) prefilter_view: TextureView,
    pub(super) _prefilter_sampler: Sampler,
    pub(super) _brdf_lut_texture: TrackedTexture,
    pub(super) brdf_lut_view: TextureView,
    pub(super) _brdf_lut_sampler: Sampler,
}

pub(super) fn create_fallback_ibl_resources(
    device: &Device,
    queue: &Queue,
) -> RenderResult<PbrIblResources> {
    let irradiance_texture = create_fallback_cube_texture(
        device,
        queue,
        "pbr_fallback_irradiance",
        [255, 255, 255, 255],
    )?;
    let prefilter_texture = create_fallback_cube_texture(
        device,
        queue,
        "pbr_fallback_prefilter",
        [255, 255, 255, 255],
    )?;
    let brdf_lut_texture =
        create_default_texture(device, queue, "pbr_fallback_brdf_lut", [255, 255, 255, 255])?;

    let irradiance_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_irradiance_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let prefilter_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_prefilter_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let brdf_lut_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_lut_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });

    Ok(PbrIblResources {
        irradiance_view: irradiance_texture.create_view(&TextureViewDescriptor {
            label: Some("pbr_fallback_irradiance_cube_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        }),
        irradiance_sampler,
        _irradiance_texture: irradiance_texture,
        prefilter_view: prefilter_texture.create_view(&TextureViewDescriptor {
            label: Some("pbr_fallback_prefilter_cube_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        }),
        _prefilter_sampler: prefilter_sampler,
        _prefilter_texture: prefilter_texture,
        brdf_lut_view: brdf_lut_texture.create_view(&TextureViewDescriptor::default()),
        _brdf_lut_sampler: brdf_lut_sampler,
        _brdf_lut_texture: brdf_lut_texture,
    })
}

fn create_fallback_cube_texture(
    device: &Device,
    queue: &Queue,
    label: &str,
    color: [u8; 4],
) -> RenderResult<TrackedTexture> {
    let texture = tracked_create_texture(
        device,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    let texels = color.repeat(6);
    queue.write_texture(
        ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texels,
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
    );
    Ok(texture)
}
