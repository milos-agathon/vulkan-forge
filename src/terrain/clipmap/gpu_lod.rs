//! P2.3: GPU LOD selection with frustum culling.
//!
//! Performs per-tile frustum culling and LOD selection on the GPU using
//! a compute shader, outputting a compact list of visible tiles.

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};

/// Configuration for GPU LOD selection.
#[derive(Debug, Clone)]
pub struct GpuLodConfig {
    /// Target pixel error budget.
    pub pixel_error_budget: f32,
    /// Viewport width in pixels.
    pub viewport_width: u32,
    /// Viewport height in pixels.
    pub viewport_height: u32,
    /// Camera field of view in radians.
    pub fov_y: f32,
    /// Maximum LOD level.
    pub max_lod: u32,
    /// Terrain width in world units.
    pub terrain_width: f32,
    /// Tile size in world units.
    pub tile_size: f32,
}

impl Default for GpuLodConfig {
    fn default() -> Self {
        Self {
            pixel_error_budget: 2.0,
            viewport_width: 1920,
            viewport_height: 1080,
            fov_y: std::f32::consts::FRAC_PI_4,
            max_lod: 4,
            terrain_width: 10000.0,
            tile_size: 256.0,
        }
    }
}

/// Uniform buffer for LOD selection parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LodSelectParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub lod_params: [f32; 4], // pixel_error_budget, viewport_height, fov_y, max_lod
    pub terrain_params: [f32; 4], // tile_size, num_tiles, variant_count, first_instance
    pub height_params: [f32; 4], // conservative world-space min/max
}

impl LodSelectParams {
    pub fn new(
        view_proj: Mat4,
        camera_pos: Vec3,
        frustum: &FrustumPlanes,
        config: &GpuLodConfig,
        num_tiles: u32,
        variant_count: u32,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
    ) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            frustum_planes: frustum.to_array(),
            lod_params: [
                config.pixel_error_budget,
                config.viewport_height as f32,
                config.fov_y,
                config.max_lod as f32,
            ],
            terrain_params: [
                config.tile_size,
                num_tiles as f32,
                variant_count as f32,
                u32::from(first_instance) as f32,
            ],
            height_params: [
                height_bounds.0,
                height_bounds.1,
                u32::from(frustum_culling) as f32,
                0.0,
            ],
        }
    }
}

/// Tile information for GPU processing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileInfo {
    pub tile_id: u32,
    pub height_min: f32,
    pub bounds_min: [f32; 2],
    pub bounds_max: [f32; 2],
    pub distance: f32,
    pub selected_lod: u32,
    pub visible: u32,
    pub height_max: f32,
}

impl TileInfo {
    pub fn new(lod: u32, x: u32, y: u32, bounds_min: Vec2, bounds_max: Vec2) -> Self {
        Self {
            tile_id: Self::pack_id(lod, x, y),
            height_min: f32::INFINITY,
            bounds_min: [bounds_min.x, bounds_min.y],
            bounds_max: [bounds_max.x, bounds_max.y],
            distance: 0.0,
            selected_lod: lod,
            visible: 1,
            height_max: f32::NEG_INFINITY,
        }
    }

    pub fn with_height_bounds(mut self, height_min: f32, height_max: f32) -> Self {
        self.height_min = height_min;
        self.height_max = height_max;
        self
    }

    pub fn pack_id(lod: u32, x: u32, y: u32) -> u32 {
        (lod << 24) | ((x & 0xFFF) << 12) | (y & 0xFFF)
    }

    pub fn unpack_id(packed: u32) -> (u32, u32, u32) {
        let lod = packed >> 24;
        let x = (packed >> 12) & 0xFFF;
        let y = packed & 0xFFF;
        (lod, x, y)
    }
}

/// Output header with atomic counters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OutputHeader {
    pub visible_count: u32,
    pub total_triangles: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Per-region indexed draw metadata consumed by the LOD compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndirectDrawTemplate {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub tile_id: u32,
}

/// WGPU `DrawIndexedIndirect` argument layout (20 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// GPU-written instance data for the compact indirect draw list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClipmapDrawInstance {
    pub transform: [[f32; 4]; 4],
    pub tile_id_lod: [u32; 2],
    pub _pad: [u32; 2],
}

impl ClipmapDrawInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Uint32x2];

    pub fn identity(tile_id: u32, lod: u32) -> Self {
        Self {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
            tile_id_lod: [tile_id, lod],
            _pad: [0; 2],
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// GPU resources kept alive until the render pass consumes the indirect list.
pub struct GpuLodDrawResources {
    pub indirect_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub instance_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub output_tiles: crate::core::resource_tracker::TrackedBuffer,
    pub output_header: crate::core::resource_tracker::TrackedBuffer,
    pub max_draw_count: u32,
    variant_count: u32,
    params: crate::core::resource_tracker::TrackedBuffer,
    _input_tiles: crate::core::resource_tracker::TrackedBuffer,
    _draw_templates: crate::core::resource_tracker::TrackedBuffer,
    bind_group: wgpu::BindGroup,
}

impl GpuLodDrawResources {
    /// Read back the compact tile list produced by the most recently submitted
    /// LOD compute pass. Queue ordering guarantees this copy observes the exact
    /// list consumed by the indirect terrain draw.
    pub(crate) fn read_selection_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> RenderResult<LodSelectionResult> {
        let header_bytes = std::mem::size_of::<OutputHeader>() as u64;
        let tile_bytes = u64::from(self.max_draw_count) * std::mem::size_of::<TileInfo>() as u64;
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("lod_selection_readback"),
                size: header_bytes + tile_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lod_selection_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.output_header, 0, &readback, 0, header_bytes);
        encoder.copy_buffer_to_buffer(&self.output_tiles, 0, &readback, header_bytes, tile_bytes);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RenderError::readback("LOD selection callback dropped"))?
            .map_err(|error| RenderError::readback(format!("LOD selection map failed: {error}")))?;
        let mapped = slice.get_mapped_range();
        let header = bytemuck::pod_read_unaligned::<OutputHeader>(
            &mapped[..std::mem::size_of::<OutputHeader>()],
        );
        let visible_count = header.visible_count.min(self.max_draw_count) as usize;
        let tile_size = std::mem::size_of::<TileInfo>();
        let visible_tiles = (0..visible_count)
            .map(|index| {
                let start = header_bytes as usize + index * tile_size;
                bytemuck::pod_read_unaligned::<TileInfo>(&mapped[start..start + tile_size])
            })
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(LodSelectionResult {
            visible_tiles,
            total_triangles: header.total_triangles,
            culled_count: self.max_draw_count - header.visible_count.min(self.max_draw_count),
        })
    }
}

/// Frustum planes for culling (Ax + By + Cz + D = 0 format).
#[derive(Debug, Clone)]
pub struct FrustumPlanes {
    pub left: Vec4,
    pub right: Vec4,
    pub bottom: Vec4,
    pub top: Vec4,
    pub near: Vec4,
    pub far: Vec4,
}

impl FrustumPlanes {
    /// Extract frustum planes from view-projection matrix.
    pub fn from_view_proj(vp: Mat4) -> Self {
        let rows = [
            Vec4::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x, vp.w_axis.x),
            Vec4::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y, vp.w_axis.y),
            Vec4::new(vp.x_axis.z, vp.y_axis.z, vp.z_axis.z, vp.w_axis.z),
            Vec4::new(vp.x_axis.w, vp.y_axis.w, vp.z_axis.w, vp.w_axis.w),
        ];

        Self {
            left: normalize_plane(rows[3] + rows[0]),
            right: normalize_plane(rows[3] - rows[0]),
            bottom: normalize_plane(rows[3] + rows[1]),
            top: normalize_plane(rows[3] - rows[1]),
            // WGPU/glam perspective matrices use a zero-to-one depth range.
            // The near clip inequality is therefore row2 >= 0, not the
            // OpenGL-style row3 + row2 >= 0.  The latter rejected every
            // clipmap tile on the NVIDIA Vulkan acceptance lane.
            near: normalize_plane(rows[2]),
            far: normalize_plane(rows[3] - rows[2]),
        }
    }

    pub fn to_array(&self) -> [[f32; 4]; 6] {
        [
            self.left.to_array(),
            self.right.to_array(),
            self.bottom.to_array(),
            self.top.to_array(),
            self.near.to_array(),
            self.far.to_array(),
        ]
    }
}

fn normalize_plane(plane: Vec4) -> Vec4 {
    let normal = plane.xyz();
    let normal_len = normal.length();
    if normal_len > 0.0 {
        plane / normal_len
    } else {
        plane
    }
}

/// GPU LOD selector using compute shaders.
pub struct GpuLodSelector {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    config: GpuLodConfig,
}

impl GpuLodSelector {
    /// Create a new GPU LOD selector.
    pub fn new(device: &wgpu::Device, config: GpuLodConfig) -> Self {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "clipmap_lod_select",
            include_str!("../../shaders/clipmap_lod_select.wgsl"),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lod_select_bind_group_layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lod_select_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("lod_select_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_main",
            },
        );

        Self {
            pipeline,
            bind_group_layout,
            config,
        }
    }

    /// Allocate the static inputs and reusable outputs for a clipmap mesh.
    pub fn create_draw_resources(
        &self,
        device: &wgpu::Device,
        tiles: &[TileInfo],
        draw_templates: &[IndirectDrawTemplate],
    ) -> RenderResult<GpuLodDrawResources> {
        let variant_count = self.config.max_lod + 1;
        if tiles.is_empty() || draw_templates.len() != tiles.len() * variant_count as usize {
            return Err(crate::core::error::RenderError::render(
                "GPU LOD selection requires every tile to have every LOD draw template",
            ));
        }

        let params = LodSelectParams::new(
            Mat4::IDENTITY,
            Vec3::ZERO,
            &FrustumPlanes::from_view_proj(Mat4::IDENTITY),
            &self.config,
            tiles.len() as u32,
            variant_count,
            false,
            (-f32::MAX, f32::MAX),
            true,
        );
        let params_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_params_buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let input_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_input_tiles"),
                contents: bytemuck::cast_slice(tiles),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let template_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_draw_templates"),
                contents: bytemuck::cast_slice(draw_templates),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;

        let output_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("lod_output_tiles"),
                size: (tiles.len() * std::mem::size_of::<TileInfo>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let indirect_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.clipmap.indirect_args"),
                size: (tiles.len() * std::mem::size_of::<DrawIndexedIndirectArgs>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let instance_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.clipmap.draw_instances"),
                size: (tiles.len() * std::mem::size_of::<ClipmapDrawInstance>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;

        let header = OutputHeader {
            visible_count: 0,
            total_triangles: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let header_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_output_header"),
                contents: bytemuck::bytes_of(&header),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lod_select_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: template_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(GpuLodDrawResources {
            indirect_buffer,
            instance_buffer,
            output_tiles: output_buffer,
            output_header: header_buffer,
            max_draw_count: tiles.len() as u32,
            variant_count,
            params: params_buffer,
            _input_tiles: input_buffer,
            _draw_templates: template_buffer,
            bind_group,
        })
    }

    /// Cull clipmap regions and write a compact indirect draw/instance list.
    pub fn encode_indirect(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuLodDrawResources,
        view_proj: Mat4,
        camera_pos: Vec3,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
    ) {
        let frustum = FrustumPlanes::from_view_proj(view_proj);
        let params = LodSelectParams::new(
            view_proj,
            camera_pos,
            &frustum,
            &self.config,
            resources.max_draw_count,
            resources.variant_count,
            first_instance,
            height_bounds,
            frustum_culling,
        );
        queue.write_buffer(&resources.params, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(&resources.output_header, 0, None);
        encoder.clear_buffer(&resources.indirect_buffer, 0, None);
        encoder.clear_buffer(&resources.instance_buffer, 0, None);
        crate::core::shader_registry::record_shader_use("clipmap_lod_select");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lod_select_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    #[cfg(test)]
    fn select_batch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tiles: &[TileInfo],
        cameras: &[(Mat4, Vec3)],
    ) -> Vec<LodSelectionResult> {
        let templates: Vec<_> = tiles
            .iter()
            .flat_map(|tile| {
                (0..=self.config.max_lod).map(move |_| IndirectDrawTemplate {
                    index_count: 3,
                    first_index: 0,
                    base_vertex: 0,
                    tile_id: tile.tile_id,
                })
            })
            .collect();
        let resources = self
            .create_draw_resources(device, tiles, &templates)
            .expect("create GPU LOD draw resources");
        let header_bytes = std::mem::size_of::<OutputHeader>() as u64;
        let tile_bytes = std::mem::size_of_val(tiles) as u64;
        let camera_stride = header_bytes + tile_bytes;
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("lod_batch_test_readback"),
                size: camera_stride * cameras.len() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .expect("create LOD batch readback");
        for (camera_index, (view_proj, camera_pos)) in cameras.iter().copied().enumerate() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lod_batch_test_encoder"),
            });
            self.encode_indirect(
                queue,
                &mut encoder,
                &resources,
                view_proj,
                camera_pos,
                false,
                (0.0, 1000.0),
                true,
            );
            let offset = camera_stride * camera_index as u64;
            encoder.copy_buffer_to_buffer(
                &resources.output_header,
                0,
                &readback,
                offset,
                header_bytes,
            );
            encoder.copy_buffer_to_buffer(
                &resources.output_tiles,
                0,
                &readback,
                offset + header_bytes,
                tile_bytes,
            );
            queue.submit(Some(encoder.finish()));
        }

        let slice = readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .expect("LOD readback callback")
            .expect("LOD readback mapping");
        let mapped = slice.get_mapped_range();
        let results = cameras
            .iter()
            .enumerate()
            .map(|(camera_index, _)| {
                let offset = camera_stride as usize * camera_index;
                let header = bytemuck::pod_read_unaligned::<OutputHeader>(
                    &mapped[offset..offset + header_bytes as usize],
                );
                let visible_count = (header.visible_count as usize).min(tiles.len());
                let tile_start = offset + header_bytes as usize;
                let tile_size = std::mem::size_of::<TileInfo>();
                let visible_tiles = (0..visible_count)
                    .map(|index| {
                        let start = tile_start + index * tile_size;
                        bytemuck::pod_read_unaligned::<TileInfo>(&mapped[start..start + tile_size])
                    })
                    .collect();
                LodSelectionResult {
                    visible_tiles,
                    total_triangles: header.total_triangles,
                    culled_count: tiles.len() as u32 - header.visible_count.min(tiles.len() as u32),
                }
            })
            .collect();
        drop(mapped);
        readback.unmap();
        results
    }
}

/// Result of GPU LOD selection.
#[derive(Debug, Clone)]
pub struct LodSelectionResult {
    /// Visible tiles after frustum culling with selected LOD.
    pub visible_tiles: Vec<TileInfo>,
    /// Total triangle count across all visible tiles.
    pub total_triangles: u32,
    /// Number of tiles culled.
    pub culled_count: u32,
}

impl LodSelectionResult {
    /// Calculate triangle reduction percentage.
    pub fn triangle_reduction(&self, full_res_triangles: u32) -> f32 {
        if full_res_triangles == 0 {
            return 0.0;
        }
        let reduction =
            (full_res_triangles as f32 - self.total_triangles as f32) / full_res_triangles as f32;
        (reduction * 100.0).max(0.0)
    }
}

/// Use the exact GPU-submitted tile/LOD list when it is available, falling
/// back to an independent CPU selection only for paths without that provenance.
pub(crate) fn prefer_submitted_tiles<F>(
    submitted_tiles: Option<&[TileInfo]>,
    cpu_fallback: F,
) -> Vec<TileInfo>
where
    F: FnOnce() -> Vec<TileInfo>,
{
    submitted_tiles.map_or_else(cpu_fallback, <[_]>::to_vec)
}

/// CPU fallback for LOD selection (used when GPU compute is unavailable).
pub fn cpu_lod_select(
    tiles: &[TileInfo],
    view_proj: Mat4,
    camera_pos: Vec3,
    config: &GpuLodConfig,
    height_bounds: (f32, f32),
) -> LodSelectionResult {
    let frustum = FrustumPlanes::from_view_proj(view_proj);
    let camera_pos_2d = Vec2::new(camera_pos.x, camera_pos.y);

    let mut visible_tiles = Vec::new();
    let mut total_triangles = 0u32;
    let mut culled_count = 0u32;

    for tile in tiles {
        let bounds_min = Vec2::from(tile.bounds_min);
        let bounds_max = Vec2::from(tile.bounds_max);
        let center = (bounds_min + bounds_max) * 0.5;

        let (height_min, height_max) = if tile.height_min <= tile.height_max {
            (tile.height_min, tile.height_max)
        } else {
            height_bounds
        };
        let visible = frustum_test_aabb(bounds_min, bounds_max, height_min, height_max, &frustum);

        if !visible {
            culled_count += 1;
            continue;
        }

        // Calculate distance and select LOD
        let distance = camera_pos_2d.distance(center);
        let selected_lod = select_lod_cpu(distance, config);

        let mut selected_tile = *tile;
        selected_tile.distance = distance;
        selected_tile.selected_lod = selected_lod;
        selected_tile.visible = 1;

        // Calculate triangle count for this LOD
        let base_triangles = 128 * 128 * 2;
        let reduction = 1u32 << (selected_lod * 2);
        total_triangles += base_triangles / reduction.max(1);

        visible_tiles.push(selected_tile);
    }

    LodSelectionResult {
        visible_tiles,
        total_triangles,
        culled_count,
    }
}

fn frustum_test_aabb(
    bounds_min: Vec2,
    bounds_max: Vec2,
    height_min: f32,
    height_max: f32,
    frustum: &FrustumPlanes,
) -> bool {
    let planes = [
        frustum.left,
        frustum.right,
        frustum.bottom,
        frustum.top,
        frustum.near,
        frustum.far,
    ];

    for plane in planes {
        let positive = Vec3::new(
            if plane.x >= 0.0 {
                bounds_max.x
            } else {
                bounds_min.x
            },
            if plane.y >= 0.0 {
                bounds_max.y
            } else {
                bounds_min.y
            },
            if plane.z >= 0.0 {
                height_max
            } else {
                height_min
            },
        );
        if plane.xyz().dot(positive) + plane.w < 0.0 {
            return false;
        }
    }
    true
}

fn select_lod_cpu(distance: f32, config: &GpuLodConfig) -> u32 {
    let safe_distance = distance.max(0.1);
    let half_fov = config.fov_y * 0.5;
    let pixels_per_unit = (config.viewport_height as f32 * 0.5) / (safe_distance * half_fov.tan());

    for lod in (0..=config.max_lod).rev() {
        let error = config.tile_size * (1 << lod) as f32 * pixels_per_unit;
        if error <= config.pixel_error_budget {
            return lod;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_selection_prefers_exact_submitted_tiles_without_recomputing() {
        let submitted = [TileInfo {
            tile_id: 97,
            height_min: -1.0,
            bounds_min: [-2.0, -3.0],
            bounds_max: [4.0, 5.0],
            distance: 12.0,
            selected_lod: 2,
            visible: 1,
            height_max: 6.0,
        }];
        let selected = prefer_submitted_tiles(Some(&submitted), || {
            panic!("CPU LOD selection must not replace submitted GPU provenance")
        });

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].tile_id, 97);
        assert_eq!(selected[0].selected_lod, 2);
    }

    #[test]
    fn test_tile_id_packing() {
        let (lod, x, y) = (3, 100, 200);
        let packed = TileInfo::pack_id(lod, x, y);
        let (l, xx, yy) = TileInfo::unpack_id(packed);
        assert_eq!((l, xx, yy), (lod, x, y));
    }

    #[test]
    fn test_frustum_planes_extraction() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 100.0, 100.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 1000.0);
        let vp = proj * view;

        let frustum = FrustumPlanes::from_view_proj(vp);

        // Planes should be normalized
        assert!((frustum.left.xyz().length() - 1.0).abs() < 0.01);
        assert!((frustum.right.xyz().length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cpu_lod_selection() {
        let tiles = vec![
            TileInfo::new(0, 0, 0, Vec2::new(0.0, 0.0), Vec2::new(256.0, 256.0)),
            TileInfo::new(0, 1, 0, Vec2::new(256.0, 0.0), Vec2::new(512.0, 256.0)),
        ];

        let view = Mat4::look_at_rh(
            Vec3::new(128.0, 100.0, 128.0),
            Vec3::new(128.0, 0.0, 128.0),
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 10000.0);
        let vp = proj * view;

        let config = GpuLodConfig::default();
        let result = cpu_lod_select(
            &tiles,
            vp,
            Vec3::new(128.0, 100.0, 128.0),
            &config,
            (0.0, 1000.0),
        );

        assert!(!result.visible_tiles.is_empty());
        assert!(result.total_triangles > 0);
    }

    #[test]
    fn test_lod_selection_distance_based() {
        let config = GpuLodConfig {
            pixel_error_budget: 2.0,
            viewport_height: 1080,
            fov_y: 45.0_f32.to_radians(),
            max_lod: 4,
            tile_size: 256.0,
            ..Default::default()
        };

        // Close distance should select low LOD (high detail)
        let lod_close = select_lod_cpu(100.0, &config);
        // Far distance should select higher LOD (lower detail)
        let lod_far = select_lod_cpu(10000.0, &config);

        assert!(lod_far >= lod_close, "Far tiles should use coarser LOD");
    }

    #[test]
    fn cpu_selection_keeps_a_tile_whose_bounds_intersect_the_frustum() {
        let tiles = vec![TileInfo::new(
            0,
            0,
            0,
            Vec2::new(0.9, -0.1),
            Vec2::new(1.5, 0.1),
        )];

        let result = cpu_lod_select(
            &tiles,
            Mat4::IDENTITY,
            Vec3::ZERO,
            &GpuLodConfig::default(),
            (0.0, 1000.0),
        );

        assert_eq!(
            result.visible_tiles.len(),
            1,
            "conservative culling must keep an AABB that crosses the frustum edge"
        );
    }

    #[test]
    fn zero_to_one_near_plane_keeps_points_in_front_of_camera() {
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let planes = FrustumPlanes::from_view_proj(projection * view);
        for plane in planes.to_array() {
            let distance = Vec3::new(plane[0], plane[1], plane[2]).dot(Vec3::ZERO) + plane[3];
            assert!(distance >= 0.0, "origin rejected by plane {plane:?}");
        }
    }

    #[test]
    #[ignore = "requires a physical GPU; the TESSELLA lane runs this exact test"]
    fn gpu_and_cpu_select_identical_tile_sets_for_1000_cameras() {
        let context = crate::core::gpu::try_ctx()
            .expect("TESSELLA GPU/CPU differential requires a physical GPU adapter");
        let config = GpuLodConfig {
            pixel_error_budget: 256.0,
            terrain_width: 2048.0,
            tile_size: 256.0,
            max_lod: 4,
            ..Default::default()
        };
        let selector = GpuLodSelector::new(&context.device, config.clone());
        let tiles: Vec<_> = (0..8)
            .flat_map(|y| {
                (0..8).map(move |x| {
                    let min = Vec2::new(-1024.0 + x as f32 * 256.0, -1024.0 + y as f32 * 256.0);
                    TileInfo::new(0, x, y, min, min + Vec2::splat(256.0))
                })
            })
            .collect();

        let mut state = 0xA511_E9B3_u32;
        let cameras: Vec<_> = (0..1000)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let angle = (state as f32 / u32::MAX as f32) * std::f32::consts::TAU;
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let radius = 900.0 + (state as f32 / u32::MAX as f32) * 900.0;
                let eye = Vec3::new(angle.cos() * radius, angle.sin() * radius, 1200.0);
                let target = Vec3::new(0.0, 0.0, 400.0);
                let view = Mat4::look_at_rh(eye, target, Vec3::Z);
                let proj = Mat4::perspective_rh(config.fov_y, 16.0 / 9.0, 1.0, 5000.0);
                (proj * view, eye)
            })
            .collect();
        let gpu = selector.select_batch_blocking(&context.device, &context.queue, &tiles, &cameras);
        let visible_sets: std::collections::BTreeSet<_> = gpu
            .iter()
            .map(|result| {
                let mut ids: Vec<_> = result
                    .visible_tiles
                    .iter()
                    .map(|tile| tile.tile_id)
                    .collect();
                ids.sort_unstable();
                ids
            })
            .collect();
        let selected_lods: std::collections::BTreeSet<_> = gpu
            .iter()
            .flat_map(|result| result.visible_tiles.iter().map(|tile| tile.selected_lod))
            .collect();
        assert!(
            visible_sets.len() > 1,
            "camera sweep must exercise multiple visible tile sets"
        );
        assert!(
            selected_lods.len() > 1,
            "camera sweep must exercise multiple selected LODs"
        );

        for (camera_index, ((view_proj, eye), gpu_result)) in
            cameras.iter().zip(gpu.iter()).enumerate()
        {
            let cpu_result = cpu_lod_select(&tiles, *view_proj, *eye, &config, (0.0, 1000.0));
            let mut cpu_ids: Vec<_> = cpu_result
                .visible_tiles
                .iter()
                .map(|tile| (tile.tile_id, tile.selected_lod))
                .collect();
            let mut gpu_ids: Vec<_> = gpu_result
                .visible_tiles
                .iter()
                .map(|tile| (tile.tile_id, tile.selected_lod))
                .collect();
            cpu_ids.sort_unstable();
            gpu_ids.sort_unstable();
            assert_eq!(gpu_ids, cpu_ids, "camera {camera_index}");
        }
    }
}
