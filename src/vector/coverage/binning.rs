use super::types::{CoverageGeometry, PrimitiveRecord, COVERAGE_TILE_SIZE};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::BufferInitDescriptor;

const COVERAGE_MEMORY_BUDGET: u64 = 512 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BinParams {
    extent_tiles: [u32; 4],
    layers_capacity: [u32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinLayout {
    pub tile_columns: u32,
    pub tile_rows: u32,
    pub tile_count: u32,
    pub layer_count: u32,
    pub tile_capacity: u32,
    pub measured_memberships: u64,
    pub allocated_index_slots: u64,
    pub allocation_bytes: u64,
}

impl BinLayout {
    pub fn measure(geometry: &CoverageGeometry) -> Result<Self, RenderError> {
        let tile_columns = geometry.width.div_ceil(COVERAGE_TILE_SIZE);
        let tile_rows = geometry.height.div_ceil(COVERAGE_TILE_SIZE);
        let tile_count = tile_columns.checked_mul(tile_rows).ok_or_else(|| {
            RenderError::Budget("vector_coverage_bin_budget: tile-count overflow".into())
        })?;
        let layer_count = u32::try_from(geometry.layers.len()).map_err(|_| {
            RenderError::Budget("vector_coverage_bin_budget: layer-count overflow".into())
        })?;
        let primitive_count = u32::try_from(geometry.primitives.len()).map_err(|_| {
            RenderError::Budget("vector_coverage_bin_budget: primitive-count overflow".into())
        })?;
        let layer_tiles = u64::from(tile_count)
            .checked_mul(u64::from(layer_count))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_bin_budget: layer-tile overflow".into())
            })?;
        let mut counts = vec![
            0_u32;
            usize::try_from(layer_tiles).map_err(|_| {
                RenderError::Budget(
                    "vector_coverage_bin_budget: layer-tile index does not fit usize".into(),
                )
            })?
        ];
        let mut measured_memberships = 0_u64;
        let extent = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
            f64::from(geometry.width),
            f64::from(geometry.height),
            0.0,
        ));

        for primitive in &geometry.primitives {
            let [min_x, min_y, max_x, max_y] = primitive.bin_bounds;
            if max_x < 0.0 || max_y < 0.0 || min_x >= extent.x || min_y >= extent.y {
                continue;
            }
            let tx0 = pixel_to_tile(min_x, tile_columns);
            let ty0 = pixel_to_tile(min_y, tile_rows);
            let tx1 = pixel_to_tile(max_x, tile_columns);
            let ty1 = pixel_to_tile(max_y, tile_rows);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    let index = u64::from(primitive.layer())
                        .checked_mul(u64::from(tile_count))
                        .and_then(|base| {
                            base.checked_add(
                                u64::from(ty) * u64::from(tile_columns) + u64::from(tx),
                            )
                        })
                        .ok_or_else(|| {
                            RenderError::Budget(
                                "vector_coverage_bin_budget: membership index overflow".into(),
                            )
                        })?;
                    let slot = &mut counts[index as usize];
                    *slot = slot.checked_add(1).ok_or_else(|| {
                        RenderError::Budget(
                            "vector_coverage_bin_budget: per-tile count overflow".into(),
                        )
                    })?;
                    measured_memberships += 1;
                }
            }
        }

        let tile_capacity = counts.into_iter().max().unwrap_or(0).max(1);
        let allocated_index_slots = layer_tiles
            .checked_mul(u64::from(tile_capacity))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_bin_budget: index-slot overflow".into())
            })?;
        let primitive_bytes = u64::from(primitive_count)
            .checked_mul(std::mem::size_of::<PrimitiveRecord>() as u64)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_bin_budget: primitive-byte overflow".into())
            })?;
        let allocation_bytes = primitive_bytes
            .max(std::mem::size_of::<PrimitiveRecord>() as u64)
            .checked_add((layer_tiles * 4).max(4))
            .and_then(|bytes| bytes.checked_add(allocated_index_slots * 4))
            .and_then(|bytes| bytes.checked_add(16))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<BinParams>() as u64))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_bin_budget: byte-count overflow".into())
            })?;
        if allocation_bytes > COVERAGE_MEMORY_BUDGET {
            return Err(RenderError::Budget(format!(
                "vector_coverage_bin_budget: required={allocation_bytes} budget={COVERAGE_MEMORY_BUDGET} \
                 memberships={measured_memberships} tile_capacity={tile_capacity}; refusing to truncate"
            )));
        }

        Ok(Self {
            tile_columns,
            tile_rows,
            tile_count,
            layer_count,
            tile_capacity,
            measured_memberships,
            allocated_index_slots,
            allocation_bytes,
        })
    }
}

fn pixel_to_tile(value: f32, tile_extent: u32) -> u32 {
    ((value.max(0.0) as u32) / COVERAGE_TILE_SIZE).min(tile_extent.saturating_sub(1))
}

pub struct CoverageBins {
    pub layout: BinLayout,
    pub primitive_buffer: TrackedBuffer,
    pub tile_counts: TrackedBuffer,
    pub tile_indices: TrackedBuffer,
    pub overflow: TrackedBuffer,
    _params: TrackedBuffer,
    bind_group: wgpu::BindGroup,
}

pub struct CoverageBinner {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CoverageBinner {
    pub fn new(device: &wgpu::Device) -> Self {
        let source = format!(
            "{}\n{}",
            include_str!("../../shaders/includes/determinism.wgsl"),
            include_str!("../../shaders/vector_coverage_bin.wgsl")
        );
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "vector_coverage_bin.wgsl",
            &source,
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Bin.BindGroupLayout"),
            entries: &[
                storage_entry(0, true),
                uniform_entry(1),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Bin.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vf.Vector.Coverage.Bin.Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        );
        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        geometry: &CoverageGeometry,
    ) -> Result<CoverageBins, RenderError> {
        let layout = BinLayout::measure(geometry)?;
        self.prepare_compiled(device, geometry, layout)
    }

    pub(super) fn prepare_compiled(
        &self,
        device: &wgpu::Device,
        geometry: &CoverageGeometry,
        layout: BinLayout,
    ) -> Result<CoverageBins, RenderError> {
        let zero_primitive = PrimitiveRecord::zeroed();
        let primitive_bytes: &[u8] = if geometry.primitives.is_empty() {
            bytemuck::bytes_of(&zero_primitive)
        } else {
            bytemuck::cast_slice(&geometry.primitives)
        };
        let primitive_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.Primitives"),
                contents: primitive_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let layer_tiles = u64::from(layout.layer_count) * u64::from(layout.tile_count);
        let tile_counts = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.Coverage.TileCounts"),
                size: (layer_tiles * 4).max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let tile_indices = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.Coverage.TileIndices"),
                size: (layout.allocated_index_slots * 4).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let overflow = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.Coverage.BinOverflow"),
                // [bin overflow, active-list overflow, breakpoint overflow,
                // dispatch-stage bitset]. Later passes share this block.
                size: 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let params_value = BinParams {
            extent_tiles: [
                geometry.width,
                geometry.height,
                layout.tile_columns,
                layout.tile_rows,
            ],
            layers_capacity: [
                layout.layer_count,
                layout.tile_capacity,
                u32::try_from(geometry.primitives.len()).map_err(|_| {
                    RenderError::Budget(
                        "vector_coverage_bin_budget: primitive-count overflow".into(),
                    )
                })?,
                COVERAGE_TILE_SIZE,
            ],
        };
        let params = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.BinParams"),
                contents: bytemuck::bytes_of(&params_value),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Coverage.Bin.BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: primitive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: overflow.as_entire_binding(),
                },
            ],
        });
        Ok(CoverageBins {
            layout,
            primitive_buffer,
            tile_counts,
            tile_indices,
            overflow,
            _params: params,
            bind_group,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bins: &CoverageBins,
        primitive_count: u32,
    ) {
        encoder.clear_buffer(&bins.tile_counts, 0, None);
        encoder.clear_buffer(&bins.overflow, 0, None);
        if primitive_count == 0 {
            return;
        }
        crate::core::shader_registry::record_shader_use("vector_coverage_bin.wgsl");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vf.Vector.Coverage.Bin.Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bins.bind_group, &[]);
        pass.dispatch_workgroups(primitive_count.div_ceil(64), 1, 1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::{PolygonDef, VectorStyle};
    use crate::vector::coverage::{CoverageGeometryBuilder, FillRule};
    use glam::Vec2;

    #[test]
    fn measured_capacity_matches_boundary_density_without_truncation() {
        let mut builder = CoverageGeometryBuilder::new(32, 16).unwrap();
        let layer = builder
            .add_layer("mosaic", FillRule::NonZero, [1.0, 1.0, 1.0, 1.0])
            .unwrap();
        for x in [0.0, 16.0] {
            builder
                .push_polygon(
                    layer,
                    &PolygonDef {
                        exterior: vec![
                            Vec2::new(x, 0.0),
                            Vec2::new(x + 16.0, 0.0),
                            Vec2::new(x + 16.0, 16.0),
                            Vec2::new(x, 16.0),
                        ],
                        holes: vec![],
                        style: VectorStyle::default(),
                    },
                )
                .unwrap();
        }
        let geometry = builder.finish().unwrap();
        let layout = BinLayout::measure(&geometry).unwrap();
        assert_eq!(layout.tile_columns, 2);
        assert_eq!(layout.tile_rows, 1);
        assert!(layout.tile_capacity >= 4);
        assert!(layout.allocated_index_slots >= layout.measured_memberships);
        assert!(layout.allocation_bytes < COVERAGE_MEMORY_BUDGET);
    }

    #[test]
    fn closed_component_edges_share_tile_membership_bounds() {
        let mut builder = CoverageGeometryBuilder::new(32, 16).unwrap();
        let layer = builder
            .add_layer("sliver", FillRule::NonZero, [1.0, 1.0, 1.0, 1.0])
            .unwrap();
        builder
            .push_polygon(
                layer,
                &PolygonDef {
                    exterior: vec![
                        Vec2::new(1.0, 3.0),
                        Vec2::new(30.0, 3.1),
                        Vec2::new(30.0, 3.2),
                        Vec2::new(1.0, 3.1),
                    ],
                    holes: vec![],
                    style: VectorStyle::default(),
                },
            )
            .unwrap();
        let geometry = builder.finish().unwrap();
        let layout = BinLayout::measure(&geometry).unwrap();

        assert!(geometry
            .primitives
            .iter()
            .all(|primitive| primitive.bin_bounds == [1.0, 3.0, 30.0, 3.2]));
        assert_eq!(layout.measured_memberships, 8);
        assert_eq!(layout.tile_capacity, 4);
    }

    #[test]
    fn bin_shader_and_pinned_math_assemble_as_valid_wgsl() {
        let source = format!(
            "{}\n{}",
            include_str!("../../shaders/includes/determinism.wgsl"),
            include_str!("../../shaders/vector_coverage_bin.wgsl")
        );
        let module =
            naga::front::wgsl::parse_str(&source).expect("combined LIMES bin shader must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("combined LIMES bin shader must validate");
    }
}
