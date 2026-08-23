use super::*;

impl HybridPathTracer {
    /// Group 0: base uniforms + lighting uniforms. Lighting lives here (not a
    /// fifth group) so every hybrid pipeline fits max_bind_groups = 4.
    pub(super) fn create_uniforms_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl0-uniforms"),
            entries: &[uniform_entry(0), uniform_entry(1)],
        })
    }

    pub(super) fn create_scene_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl1-scene"),
            entries: &[
                storage_entry(0, true),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        })
    }

    pub(super) fn create_accum_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl2-accum"),
            entries: &[
                storage_entry(0, false),
                // PROMETHEUS terrain resources (dummy-bound for non-terrain
                // renders): DEM height texture, min-max pyramid, terrain
                // uniforms, Welford variance buffer, current-frame ReSTIR
                // candidate reservoirs, env map, and the merged (prev)
                // reservoirs from the temporal+spatial reuse chain.
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                uniform_entry(3),
                storage_entry(4, false),
                storage_entry(5, false),
                sampled_texture_entry(6),
                storage_entry(7, false),
                uniform_entry(10),
            ],
        })
    }

    pub(super) fn create_output_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl3-out"),
            entries: &[
                texture_entry(0, wgpu::TextureFormat::Rgba16Float),
                texture_entry(1, wgpu::TextureFormat::Rgba16Float),
                texture_entry(2, wgpu::TextureFormat::Rgba16Float),
                texture_entry(3, wgpu::TextureFormat::R32Float),
                texture_entry(4, wgpu::TextureFormat::Rgba16Float),
                texture_entry(5, wgpu::TextureFormat::Rgba16Float),
                texture_entry(6, wgpu::TextureFormat::Rgba16Float),
                texture_entry(7, wgpu::TextureFormat::Rgba8Unorm),
            ],
        })
    }

    /// Group 2 variant for the `main_terrain_gbuffer` entry: terrain
    /// textures/uniforms plus the ReSTIR G-buffer the spatial pass reads.
    /// Kept out of the accum layout so the main kernels stay within 8 storage
    /// buffers per stage.
    pub(super) fn create_terrain_gbuffer_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl2-terrain-gbuffer"),
            entries: &[
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                uniform_entry(3),
                storage_entry(8, false),
                storage_entry(9, false),
                uniform_entry(10),
            ],
        })
    }

    /// Group 2 of pt_restir_temporal.wgsl: prev, curr -> out.
    pub(super) fn create_restir_temporal_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl-restir-temporal"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        })
    }

    /// Group 1 of pt_restir_spatial.wgsl: scene lights + ReSTIR G-buffer.
    pub(super) fn create_restir_spatial_scene_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl-restir-spatial-scene"),
            entries: &[
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(10, true),
                storage_entry(11, true),
            ],
        })
    }

    /// Group 2 of pt_restir_spatial.wgsl: in -> out reservoirs.
    pub(super) fn create_restir_spatial_reuse_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl-restir-spatial-reuse"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        })
    }

    /// Placeholder for pipeline-layout group slots a shader never touches
    /// (e.g. group 1 of the temporal pass).
    pub(super) fn create_empty_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl-empty"),
            entries: &[],
        })
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
