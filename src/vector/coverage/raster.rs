use super::binning::CoverageBins;
use super::math::{primitive_active, primitive_x};
use super::types::{CoverageGeometry, FillRule};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::BufferInitDescriptor;

const COVERAGE_MEMORY_BUDGET: u64 = 512 * 1024 * 1024;
const MAX_ACTIVE_PRIMITIVES: u32 = 96;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RasterParams {
    extent_layers: [u32; 4],
    limits: [u32; 4],
    dispatch: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerRuleRecord {
    values: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelDispatchRecord {
    /// [layer_pixel_offset, component-list offset, component count, reserved].
    values: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComponentRecord {
    /// [first primitive, primitive count, layer, simple-stroke flag].
    values: [u32; 4],
}

struct PixelDispatchLists {
    dispatch: Vec<PixelDispatchRecord>,
    simple_dispatch_count: u32,
    component_indices: Vec<u32>,
    components: Vec<ComponentRecord>,
    resolve_pixels: Vec<u32>,
}

pub(super) struct CoverageRasterPlan {
    baselines: Vec<i32>,
    rules: Vec<LayerRuleRecord>,
    pixel_lists: PixelDispatchLists,
}

impl CoverageRasterPlan {
    pub(super) fn compile(geometry: &CoverageGeometry) -> Result<Self, RenderError> {
        let baselines = build_pixel_baselines(geometry);
        let rules = geometry
            .layers
            .iter()
            .map(|layer| LayerRuleRecord {
                values: [layer.fill_rule as u32, 0, 0, 0],
            })
            .collect();
        let pixel_lists = build_pixel_dispatch_lists(geometry)?;
        Ok(Self {
            baselines,
            rules,
            pixel_lists,
        })
    }
}

pub struct CoverageRasterResources {
    pub coverage: TrackedBuffer,
    pub coverage_bytes: u64,
    pub allocation_bytes: u64,
    pub active_pixel_count: u32,
    pub simple_pixel_count: u32,
    pub generic_pixel_count: u32,
    pub resolve_pixel_count: u32,
    pub resolve_pixels: TrackedBuffer,
    _baselines: TrackedBuffer,
    _dispatch: TrackedBuffer,
    _component_indices: TrackedBuffer,
    _components: TrackedBuffer,
    _rules: TrackedBuffer,
    _simple_params: TrackedBuffer,
    _generic_params: TrackedBuffer,
    simple_bind_group: wgpu::BindGroup,
    generic_bind_group: wgpu::BindGroup,
}

pub struct CoverageRasterizer {
    generic_pipeline: wgpu::ComputePipeline,
    simple_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CoverageRasterizer {
    pub fn new(device: &wgpu::Device) -> Self {
        let source = raster_shader_source();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "vector_coverage_raster.wgsl",
            &source,
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Raster.BindGroupLayout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                uniform_entry(3),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Coverage.Raster.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let generic_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vf.Vector.Coverage.Raster.Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        );
        let simple_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vf.Vector.Coverage.Raster.SimpleCapsulePipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main_simple_capsule",
            },
        );
        Self {
            generic_pipeline,
            simple_pipeline,
            bind_group_layout,
        }
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        geometry: &CoverageGeometry,
        bins: &CoverageBins,
    ) -> Result<CoverageRasterResources, RenderError> {
        let plan = CoverageRasterPlan::compile(geometry)?;
        self.prepare_compiled(device, geometry, bins, &plan)
    }

    pub(super) fn prepare_compiled(
        &self,
        device: &wgpu::Device,
        geometry: &CoverageGeometry,
        bins: &CoverageBins,
        plan: &CoverageRasterPlan,
    ) -> Result<CoverageRasterResources, RenderError> {
        if bins.layout.layer_count as usize != geometry.layers.len() {
            return Err(RenderError::Upload(
                "vector_coverage_raster_layout: bin and geometry layer counts differ".into(),
            ));
        }
        let baselines = &plan.baselines;
        let pixel_lists = &plan.pixel_lists;
        let rules = &plan.rules;
        let pixel_count = u64::from(geometry.width)
            .checked_mul(u64::from(geometry.height))
            .and_then(|count| count.checked_mul(u64::from(bins.layout.layer_count)))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: pixel-count overflow".into())
            })?;
        let coverage_bytes = pixel_count.checked_mul(4).ok_or_else(|| {
            RenderError::Budget("vector_coverage_raster_budget: coverage-byte overflow".into())
        })?;
        let baseline_bytes = (baselines.len() as u64).checked_mul(4).ok_or_else(|| {
            RenderError::Budget("vector_coverage_raster_budget: baseline-byte overflow".into())
        })?;
        let rule_bytes = (rules.len() as u64)
            .checked_mul(std::mem::size_of::<LayerRuleRecord>() as u64)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: rule-byte overflow".into())
            })?;
        let dispatch_bytes = (pixel_lists.dispatch.len() as u64)
            .checked_mul(std::mem::size_of::<PixelDispatchRecord>() as u64)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: dispatch-byte overflow".into())
            })?;
        let component_index_bytes = (pixel_lists.component_indices.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| {
                RenderError::Budget(
                    "vector_coverage_raster_budget: component-index overflow".into(),
                )
            })?;
        let component_bytes = (pixel_lists.components.len() as u64)
            .checked_mul(std::mem::size_of::<ComponentRecord>() as u64)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: component-byte overflow".into())
            })?;
        let resolve_pixel_bytes = (pixel_lists.resolve_pixels.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: resolve-pixel overflow".into())
            })?;
        let total_bytes = bins
            .layout
            .allocation_bytes
            .checked_add(coverage_bytes)
            .and_then(|bytes| bytes.checked_add(baseline_bytes))
            .and_then(|bytes| bytes.checked_add(rule_bytes))
            .and_then(|bytes| bytes.checked_add(dispatch_bytes.max(16)))
            .and_then(|bytes| bytes.checked_add(component_index_bytes.max(4)))
            .and_then(|bytes| bytes.checked_add(component_bytes.max(16)))
            .and_then(|bytes| bytes.checked_add(resolve_pixel_bytes.max(4)))
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<RasterParams>() as u64))
            .ok_or_else(|| {
                RenderError::Budget("vector_coverage_raster_budget: total-byte overflow".into())
            })?;
        if total_bytes > COVERAGE_MEMORY_BUDGET {
            return Err(RenderError::Budget(format!(
                "vector_coverage_raster_budget: required={total_bytes} \
                 budget={COVERAGE_MEMORY_BUDGET}; refusing to truncate"
            )));
        }

        let baseline_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.TileBaselines"),
                contents: bytemuck::cast_slice(baselines),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let rule_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.LayerRules"),
                contents: bytemuck::cast_slice(rules),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let zero_dispatch = PixelDispatchRecord::zeroed();
        let dispatch_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.PixelDispatch"),
                contents: if pixel_lists.dispatch.is_empty() {
                    bytemuck::bytes_of(&zero_dispatch)
                } else {
                    bytemuck::cast_slice(&pixel_lists.dispatch)
                },
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let zero_component_index = 0_u32;
        let component_index_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.PixelComponentIndices"),
                contents: if pixel_lists.component_indices.is_empty() {
                    bytemuck::bytes_of(&zero_component_index)
                } else {
                    bytemuck::cast_slice(&pixel_lists.component_indices)
                },
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let zero_component = ComponentRecord::zeroed();
        let component_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.Components"),
                contents: if pixel_lists.components.is_empty() {
                    bytemuck::bytes_of(&zero_component)
                } else {
                    bytemuck::cast_slice(&pixel_lists.components)
                },
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let zero_resolve_pixel = 0_u32;
        let resolve_pixel_buffer = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.ResolvePixels"),
                contents: if pixel_lists.resolve_pixels.is_empty() {
                    bytemuck::bytes_of(&zero_resolve_pixel)
                } else {
                    bytemuck::cast_slice(&pixel_lists.resolve_pixels)
                },
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;
        let active_pixel_count = u32::try_from(pixel_lists.dispatch.len()).map_err(|_| {
            RenderError::Budget(
                "vector_coverage_raster_budget: active-pixel count exceeds u32".into(),
            )
        })?;
        let simple_pixel_count = pixel_lists.simple_dispatch_count;
        let generic_pixel_count = active_pixel_count
            .checked_sub(simple_pixel_count)
            .ok_or_else(|| {
                RenderError::Render(
                    "vector_coverage_raster_internal: simple dispatch count exceeds total".into(),
                )
            })?;
        let resolve_pixel_count =
            u32::try_from(pixel_lists.resolve_pixels.len()).map_err(|_| {
                RenderError::Budget(
                    "vector_coverage_raster_budget: resolve-pixel count exceeds u32".into(),
                )
            })?;
        let pixels_per_layer = u32::try_from(
            u64::from(geometry.width)
                .checked_mul(u64::from(geometry.height))
                .ok_or_else(|| {
                    RenderError::Budget(
                        "vector_coverage_raster_budget: pixel-count overflow".into(),
                    )
                })?,
        )
        .map_err(|_| {
            RenderError::Budget(
                "vector_coverage_raster_budget: pixels per layer exceeds u32".into(),
            )
        })?;
        let common_extent = [
            geometry.width,
            geometry.height,
            bins.layout.layer_count,
            pixels_per_layer,
        ];
        let simple_params_value = RasterParams {
            extent_layers: common_extent,
            limits: [6, 40, 0, 0],
            dispatch: [simple_pixel_count, 0, 0, 0],
        };
        let generic_params_value = RasterParams {
            extent_layers: common_extent,
            limits: [MAX_ACTIVE_PRIMITIVES, 0, 0, 0],
            dispatch: [generic_pixel_count, simple_pixel_count, 0, 0],
        };
        let simple_params = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.SimpleRasterParams"),
                contents: bytemuck::bytes_of(&simple_params_value),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let generic_params = tracked_create_buffer_init(
            device,
            &BufferInitDescriptor {
                label: Some("vf.Vector.Coverage.GenericRasterParams"),
                contents: bytemuck::bytes_of(&generic_params_value),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let coverage = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.Coverage.Output"),
                size: coverage_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let make_bind_group = |label, params: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout,
                entries: &[
                    entry(0, &bins.primitive_buffer),
                    entry(1, &baseline_buffer),
                    entry(2, &rule_buffer),
                    entry(3, params),
                    entry(4, &coverage),
                    entry(5, &bins.overflow),
                    entry(6, &dispatch_buffer),
                    entry(7, &component_index_buffer),
                    entry(8, &component_buffer),
                ],
            })
        };
        let simple_bind_group = make_bind_group(
            "vf.Vector.Coverage.Raster.SimpleCapsuleBindGroup",
            &simple_params,
        );
        let generic_bind_group = make_bind_group(
            "vf.Vector.Coverage.Raster.GenericBindGroup",
            &generic_params,
        );
        Ok(CoverageRasterResources {
            coverage,
            coverage_bytes,
            allocation_bytes: total_bytes,
            active_pixel_count,
            simple_pixel_count,
            generic_pixel_count,
            resolve_pixel_count,
            resolve_pixels: resolve_pixel_buffer,
            _baselines: baseline_buffer,
            _dispatch: dispatch_buffer,
            _component_indices: component_index_buffer,
            _components: component_buffer,
            _rules: rule_buffer,
            _simple_params: simple_params,
            _generic_params: generic_params,
            simple_bind_group,
            generic_bind_group,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &CoverageRasterResources,
        _geometry: &CoverageGeometry,
    ) {
        encoder.clear_buffer(&resources.coverage, 0, None);
        if resources.active_pixel_count == 0 {
            return;
        }
        crate::core::shader_registry::record_shader_use("vector_coverage_raster.wgsl");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vf.Vector.Coverage.Raster.Pass"),
            timestamp_writes: None,
        });
        if resources.simple_pixel_count != 0 {
            pass.set_pipeline(&self.simple_pipeline);
            pass.set_bind_group(0, &resources.simple_bind_group, &[]);
            pass.dispatch_workgroups(resources.simple_pixel_count.div_ceil(64), 1, 1);
        }
        if resources.generic_pixel_count != 0 {
            pass.set_pipeline(&self.generic_pipeline);
            pass.set_bind_group(0, &resources.generic_bind_group, &[]);
            pass.dispatch_workgroups(resources.generic_pixel_count.div_ceil(64), 1, 1);
        }
    }
}

fn build_pixel_baselines(geometry: &CoverageGeometry) -> Vec<i32> {
    let row_count = geometry.layers.len() * geometry.height as usize;
    let mut rows: Vec<Vec<(f64, i32, u32)>> = vec![Vec::new(); row_count];
    let height = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
        f64::from(geometry.height),
        0.0,
        0.0,
    ))
    .x;
    for primitive in &geometry.primitives {
        let min_row = primitive.bounds[1].floor().max(0.0) as u32;
        let max_row = primitive.bounds[3].ceil().max(0.0).min(height) as u32;
        for row in min_row..max_row {
            let y = f64::from(row) + 0.5;
            if primitive_active(primitive, y) {
                rows[primitive.layer() as usize * geometry.height as usize + row as usize].push((
                    primitive_x(primitive, y),
                    primitive.winding(),
                    primitive.stable_id(),
                ));
            }
        }
    }

    let mut result =
        vec![0_i32; geometry.layers.len() * geometry.height as usize * geometry.width as usize];
    for (row_index, crossings) in rows.iter_mut().enumerate() {
        crossings.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.2.cmp(&right.2))
        });
        let layer = row_index / geometry.height as usize;
        let rule = geometry.layers[layer].fill_rule;
        let row = row_index % geometry.height as usize;
        let mut cursor = 0;
        let mut state = 0_i32;
        for pixel_x in 0..geometry.width {
            let boundary_x = f64::from(pixel_x);
            while cursor < crossings.len() && crossings[cursor].0 < boundary_x {
                state += match rule {
                    FillRule::NonZero => crossings[cursor].1,
                    FillRule::EvenOdd => 1,
                };
                cursor += 1;
            }
            result[(layer * geometry.height as usize + row) * geometry.width as usize
                + pixel_x as usize] = state;
        }
    }
    result
}

fn build_pixel_dispatch_lists(
    geometry: &CoverageGeometry,
) -> Result<PixelDispatchLists, RenderError> {
    let pixels_per_layer = u64::from(geometry.width) * u64::from(geometry.height);
    let layer_count = u64::try_from(geometry.layers.len()).map_err(|_| {
        RenderError::Budget("vector_coverage_raster_budget: layer count exceeds u64".into())
    })?;
    let total_pixels = pixels_per_layer.checked_mul(layer_count).ok_or_else(|| {
        RenderError::Budget("vector_coverage_raster_budget: active-mask overflow".into())
    })?;
    let mut active = vec![
        false;
        usize::try_from(total_pixels).map_err(|_| {
            RenderError::Budget("vector_coverage_raster_budget: active-mask exceeds usize".into())
        })?
    ];
    let mut component_counts = vec![0_u32; active.len()];
    let mut primitive_counts = vec![0_u32; active.len()];
    let screen_pixel_count = usize::try_from(pixels_per_layer).map_err(|_| {
        RenderError::Budget("vector_coverage_raster_budget: screen mask exceeds usize".into())
    })?;
    let mut resolve_active = vec![false; screen_pixel_count];
    let extent = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
        f64::from(geometry.width),
        f64::from(geometry.height),
        0.0,
    ));

    let mut components = Vec::new();
    let mut first = 0_usize;
    while first < geometry.primitives.len() {
        let primitive = geometry.primitives[first];
        let key = (primitive.layer(), primitive.bin_bounds.map(f32::to_bits));
        let mut end = first + 1;
        while end < geometry.primitives.len() {
            let candidate = geometry.primitives[end];
            if (candidate.layer(), candidate.bin_bounds.map(f32::to_bits)) != key {
                break;
            }
            end += 1;
        }
        let first_primitive = u32::try_from(first).map_err(|_| {
            RenderError::Budget(
                "vector_coverage_raster_budget: primitive offset exceeds u32".into(),
            )
        })?;
        let primitive_count = u32::try_from(end - first).map_err(|_| {
            RenderError::Budget("vector_coverage_raster_budget: component size exceeds u32".into())
        })?;
        let component_primitives = &geometry.primitives[first..end];
        let line_count = component_primitives
            .iter()
            .filter(|record| record.kind() == super::types::PrimitiveKind::Line)
            .count();
        let arc_count = component_primitives
            .iter()
            .filter(|record| record.kind() == super::types::PrimitiveKind::Arc)
            .count();
        let is_simple_stroke = component_primitives.len() == 6 && line_count == 2 && arc_count == 4;
        components.push(ComponentRecord {
            values: [
                first_primitive,
                primitive_count,
                primitive.layer(),
                u32::from(is_simple_stroke),
            ],
        });
        first = end;
    }

    for component in &components {
        let primitive = geometry.primitives[component.values[0] as usize];
        let [min_x, min_y, max_x, max_y] = primitive.bin_bounds;
        let x0 = min_x.floor().max(0.0).min(extent.x) as u32;
        let y0 = min_y.floor().max(0.0).min(extent.y) as u32;
        let x1 = max_x.ceil().max(0.0).min(extent.x) as u32;
        let y1 = max_y.ceil().max(0.0).min(extent.y) as u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let offset = u64::from(primitive.layer()) * pixels_per_layer
                    + u64::from(y) * u64::from(geometry.width)
                    + u64::from(x);
                let offset = usize::try_from(offset).map_err(|_| {
                    RenderError::Budget(
                        "vector_coverage_raster_budget: active offset exceeds usize".into(),
                    )
                })?;
                active[offset] = true;
                resolve_active[y as usize * geometry.width as usize + x as usize] = true;
                component_counts[offset] =
                    component_counts[offset].checked_add(1).ok_or_else(|| {
                        RenderError::Budget(
                            "vector_coverage_raster_budget: component-list overflow".into(),
                        )
                    })?;
                primitive_counts[offset] = primitive_counts[offset]
                    .checked_add(component.values[1])
                    .ok_or_else(|| {
                        RenderError::Budget(
                            "vector_coverage_raster_budget: active-list overflow".into(),
                        )
                    })?;
                if primitive_counts[offset] > MAX_ACTIVE_PRIMITIVES {
                    return Err(RenderError::Budget(format!(
                        "{{\"status\":\"vector_coverage_active_list_overflow\",\
                         \"pixel_offset\":{offset},\"required\":{},\"capacity\":{}}}",
                        primitive_counts[offset], MAX_ACTIVE_PRIMITIVES
                    )));
                }
            }
        }
    }

    let mut dispatch = Vec::new();
    let mut dispatch_by_pixel = vec![u32::MAX; active.len()];
    let mut component_offset = 0_u64;
    for (pixel_offset, is_active) in active.into_iter().enumerate() {
        if !is_active {
            continue;
        }
        let component_count = component_counts[pixel_offset];
        let dispatch_index = u32::try_from(dispatch.len()).map_err(|_| {
            RenderError::Budget("vector_coverage_raster_budget: dispatch index exceeds u32".into())
        })?;
        dispatch_by_pixel[pixel_offset] = dispatch_index;
        dispatch.push(PixelDispatchRecord {
            values: [
                u32::try_from(pixel_offset).map_err(|_| {
                    RenderError::Budget(
                        "vector_coverage_raster_budget: active-pixel index exceeds u32".into(),
                    )
                })?,
                u32::try_from(component_offset).map_err(|_| {
                    RenderError::Budget(
                        "vector_coverage_raster_budget: component offset exceeds u32".into(),
                    )
                })?,
                component_count,
                0,
            ],
        });
        component_offset = component_offset
            .checked_add(u64::from(component_count))
            .ok_or_else(|| {
                RenderError::Budget(
                    "vector_coverage_raster_budget: component-list size overflow".into(),
                )
            })?;
    }
    let mut component_indices = vec![
        0_u32;
        usize::try_from(component_offset).map_err(|_| {
            RenderError::Budget(
                "vector_coverage_raster_budget: component-list exceeds usize".into(),
            )
        })?
    ];
    let mut cursors = dispatch
        .iter()
        .map(|record| record.values[1])
        .collect::<Vec<_>>();
    for (component_index, component) in components.iter().enumerate() {
        let primitive = geometry.primitives[component.values[0] as usize];
        let [min_x, min_y, max_x, max_y] = primitive.bin_bounds;
        let x0 = min_x.floor().max(0.0).min(extent.x) as u32;
        let y0 = min_y.floor().max(0.0).min(extent.y) as u32;
        let x1 = max_x.ceil().max(0.0).min(extent.x) as u32;
        let y1 = max_y.ceil().max(0.0).min(extent.y) as u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let pixel_offset = u64::from(primitive.layer()) * pixels_per_layer
                    + u64::from(y) * u64::from(geometry.width)
                    + u64::from(x);
                let pixel_offset = usize::try_from(pixel_offset).map_err(|_| {
                    RenderError::Budget(
                        "vector_coverage_raster_budget: active offset exceeds usize".into(),
                    )
                })?;
                let dispatch_index = dispatch_by_pixel[pixel_offset];
                if dispatch_index == u32::MAX {
                    return Err(RenderError::Render(
                        "vector_coverage_raster_internal: missing active dispatch".into(),
                    ));
                }
                let cursor = &mut cursors[dispatch_index as usize];
                component_indices[*cursor as usize] =
                    u32::try_from(component_index).map_err(|_| {
                        RenderError::Budget(
                            "vector_coverage_raster_budget: component index exceeds u32".into(),
                        )
                    })?;
                *cursor = cursor.checked_add(1).ok_or_else(|| {
                    RenderError::Budget(
                        "vector_coverage_raster_budget: component cursor overflow".into(),
                    )
                })?;
            }
        }
    }
    let mut simple_dispatch = Vec::new();
    let mut generic_dispatch = Vec::new();
    for record in dispatch {
        let is_simple = record.values[2] == 1
            && components[component_indices[record.values[1] as usize] as usize].values[3] != 0;
        if is_simple {
            simple_dispatch.push(record);
        } else {
            generic_dispatch.push(record);
        }
    }
    let simple_dispatch_count = u32::try_from(simple_dispatch.len()).map_err(|_| {
        RenderError::Budget(
            "vector_coverage_raster_budget: simple dispatch count exceeds u32".into(),
        )
    })?;
    simple_dispatch.extend(generic_dispatch);
    let dispatch = simple_dispatch;
    let resolve_pixels = resolve_active
        .into_iter()
        .enumerate()
        .filter_map(|(index, is_active)| is_active.then_some(index))
        .map(|index| {
            u32::try_from(index).map_err(|_| {
                RenderError::Budget(
                    "vector_coverage_raster_budget: resolve-pixel index exceeds u32".into(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PixelDispatchLists {
        dispatch,
        simple_dispatch_count,
        component_indices,
        components,
        resolve_pixels,
    })
}

fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
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

pub(super) fn raster_shader_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../../shaders/includes/determinism.wgsl"),
        include_str!("../../shaders/vector_coverage_raster.wgsl")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::{PolygonDef, PolylineDef, VectorStyle};
    use crate::vector::coverage::CoverageGeometryBuilder;
    use glam::Vec2;

    #[test]
    fn pixel_dispatch_is_stable_and_screen_resolve_is_deduplicated() {
        let mut builder = CoverageGeometryBuilder::new(8, 8).unwrap();
        let layer = builder
            .add_layer("square", FillRule::NonZero, [1.0; 4])
            .unwrap();
        builder
            .push_polygon(
                layer,
                &PolygonDef {
                    exterior: vec![
                        Vec2::new(1.0, 1.0),
                        Vec2::new(3.0, 1.0),
                        Vec2::new(3.0, 3.0),
                        Vec2::new(1.0, 3.0),
                    ],
                    holes: vec![],
                    style: VectorStyle::default(),
                },
            )
            .unwrap();
        let lists = build_pixel_dispatch_lists(&builder.finish().unwrap()).unwrap();

        assert_eq!(
            lists
                .dispatch
                .iter()
                .map(|record| record.values[0])
                .collect::<Vec<_>>(),
            vec![9, 10, 17, 18]
        );
        assert_eq!(lists.component_indices, vec![0; 4]);
        assert_eq!(lists.resolve_pixels, vec![9, 10, 17, 18]);
        assert_eq!(lists.components[0].values[3], 0);
        assert_eq!(lists.simple_dispatch_count, 0);
    }

    #[test]
    fn round_capsule_records_use_the_simple_component_fast_path() {
        let mut builder = CoverageGeometryBuilder::new(8, 8).unwrap();
        let layer = builder
            .add_layer("road", FillRule::NonZero, [1.0; 4])
            .unwrap();
        builder
            .push_round_polyline(
                layer,
                &PolylineDef {
                    path: vec![Vec2::new(1.0, 1.0), Vec2::new(3.0, 1.0)],
                    style: VectorStyle {
                        stroke_width: 0.5,
                        ..VectorStyle::default()
                    },
                },
            )
            .unwrap();
        let lists = build_pixel_dispatch_lists(&builder.finish().unwrap()).unwrap();

        assert_eq!(lists.components.len(), 1);
        assert_eq!(lists.components[0].values[1], 6);
        assert_eq!(lists.components[0].values[3], 1);
        assert_eq!(lists.dispatch.len(), 8);
        assert_eq!(lists.simple_dispatch_count, 8);
        assert_eq!(lists.component_indices, vec![0; 8]);
        assert_eq!(lists.resolve_pixels.len(), 8);
    }

    #[test]
    fn coincident_capsules_do_not_take_the_single_boundary_fast_path() {
        let mut builder = CoverageGeometryBuilder::new(8, 8).unwrap();
        let layer = builder
            .add_layer("duplicate-road", FillRule::NonZero, [1.0; 4])
            .unwrap();
        let line = PolylineDef {
            path: vec![Vec2::new(1.0, 1.0), Vec2::new(3.0, 1.0)],
            style: VectorStyle {
                stroke_width: 0.5,
                ..VectorStyle::default()
            },
        };
        builder.push_round_polyline(layer, &line).unwrap();
        builder.push_round_polyline(layer, &line).unwrap();
        let lists = build_pixel_dispatch_lists(&builder.finish().unwrap()).unwrap();

        assert_eq!(lists.components.len(), 1);
        assert_eq!(lists.components[0].values[3], 0);
        assert_eq!(lists.simple_dispatch_count, 0);
    }
}
