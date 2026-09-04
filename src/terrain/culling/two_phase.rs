use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use crate::core::screen_space_effects::HzbPyramid;
use crate::terrain::clipmap::gpu_lod::{
    ClipmapDrawInstance, DrawIndexedIndirectArgs, GpuLodDrawResources,
};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::sync::{Mutex, OnceLock};

/// The compute source both the production pipeline and the shader-differential
/// test compile, so neither can drift away from the other.
const HZB_CULL_SOURCE: &str = include_str!("../../shaders/hzb_cull.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    view_proj: [[f32; 4]; 4],
    height_bounds: [f32; 4],
    viewport: [f32; 4],
    control: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct CullHeader {
    draw_count: u32,
    rejected_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullingStats {
    pub frustum_passing: u32,
    pub phase1_drawn: u32,
    pub phase1_rejected: u32,
    pub phase2_recovered: u32,
    pub final_drawn: u32,
    pub culled: u32,
    pub projection_bypassed: u32,
    pub background_bypassed: u32,
}

static LAST_STATS: OnceLock<Mutex<CullingStats>> = OnceLock::new();

pub fn publish_stats(stats: CullingStats) {
    if let Ok(mut current) = LAST_STATS
        .get_or_init(|| Mutex::new(CullingStats::default()))
        .lock()
    {
        *current = stats;
    }
}

pub fn latest_stats() -> CullingStats {
    LAST_STATS
        .get_or_init(|| Mutex::new(CullingStats::default()))
        .lock()
        .map(|stats| *stats)
        .unwrap_or_default()
}

pub struct CullDrawResources {
    pub indirect_buffer: TrackedBuffer,
    pub instance_buffer: TrackedBuffer,
    pub max_draw_count: u32,
    header: TrackedBuffer,
}

pub struct TwoPhaseTerrainCuller {
    pyramids: [HzbPyramid; 2],
    previous_index: usize,
    previous_valid: bool,
    previous_view_proj: Mat4,
    pipeline: wgpu::ComputePipeline,
    phase1_params: TrackedBuffer,
    phase2_params: TrackedBuffer,
    phase1_bind_groups: [wgpu::BindGroup; 2],
    phase2_bind_groups: [wgpu::BindGroup; 2],
    phase1: CullDrawResources,
    phase2: CullDrawResources,
    rejected_indices: TrackedBuffer,
    stats_readback: TrackedBuffer,
    width: u32,
    height: u32,
    mip_bias: u32,
    max_mip: u32,
    stats: CullingStats,
}

fn terrain_hzb_extent(extent: u32) -> u32 {
    (extent / 2).max(1)
}

impl TwoPhaseTerrainCuller {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        input: &GpuLodDrawResources,
    ) -> RenderResult<Self> {
        // The terrain pyramid begins at the full-resolution depth buffer's mip
        // 1. The initial HZB pass therefore fuses the depth copy and first MAX
        // reduction instead of allocating or writing a full-resolution R32F mip.
        let hzb_width = terrain_hzb_extent(width);
        let hzb_height = terrain_hzb_extent(height);
        let pyramids = [
            HzbPyramid::new_max_reduced(device, hzb_width, hzb_height)?,
            HzbPyramid::new_max_reduced(device, hzb_width, hzb_height)?,
        ];
        let mip_bias = u32::from(width > 1 || height > 1);
        let max_mip = pyramids[0].mip_count.saturating_sub(1);
        let layout = create_cull_layout(device);
        let pipeline = create_cull_pipeline(device, &layout);
        let zero_params = CullParams::zeroed();
        let phase1_params = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.hzb_cull.phase1_params"),
                contents: bytemuck::bytes_of(&zero_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let phase2_params = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.hzb_cull.phase2_params"),
                contents: bytemuck::bytes_of(&zero_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let phase1 = create_draw_resources(device, input.max_draw_count, "phase1")?;
        let phase2 = create_draw_resources(device, input.max_draw_count, "phase2")?;
        let rejected_indices = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.hzb_cull.rejected_indices"),
                size: u64::from(input.max_draw_count) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let stats_readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.hzb_cull.stats_readback"),
                size: 48,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let pyramid_views = [pyramids[0].texture_view(), pyramids[1].texture_view()];
        let phase1_bind_groups = [
            create_bind_group(
                device,
                &layout,
                &phase1_params,
                &pyramid_views[0],
                &input.output_header,
                input,
                &phase1,
                &rejected_indices,
                "phase1.previous0",
            ),
            create_bind_group(
                device,
                &layout,
                &phase1_params,
                &pyramid_views[1],
                &input.output_header,
                input,
                &phase1,
                &rejected_indices,
                "phase1.previous1",
            ),
        ];
        let phase2_bind_groups = [
            create_bind_group(
                device,
                &layout,
                &phase2_params,
                &pyramid_views[0],
                &phase1.header,
                input,
                &phase2,
                &rejected_indices,
                "phase2.fresh0",
            ),
            create_bind_group(
                device,
                &layout,
                &phase2_params,
                &pyramid_views[1],
                &phase1.header,
                input,
                &phase2,
                &rejected_indices,
                "phase2.fresh1",
            ),
        ];
        Ok(Self {
            pyramids,
            previous_index: 0,
            previous_valid: false,
            previous_view_proj: Mat4::IDENTITY,
            pipeline,
            phase1_params,
            phase2_params,
            phase1_bind_groups,
            phase2_bind_groups,
            phase1,
            phase2,
            rejected_indices,
            stats_readback,
            width,
            height,
            mip_bias,
            max_mip,
            stats: CullingStats::default(),
        })
    }

    pub fn phase1(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_view_proj: Mat4,
        height_bounds: (f32, f32),
        first_instance: bool,
    ) -> &CullDrawResources {
        let view_proj = if self.previous_valid {
            self.previous_view_proj
        } else {
            current_view_proj
        };
        let params = self.params(
            view_proj,
            height_bounds,
            1,
            self.previous_valid,
            first_instance,
        );
        queue.write_buffer(&self.phase1_params, 0, bytemuck::bytes_of(&params));
        clear_outputs(encoder, &self.phase1, Some(&self.rejected_indices));
        self.dispatch(encoder, &self.phase1_bind_groups[self.previous_index]);
        &self.phase1
    }

    /// Build the pyramid phase 2 tests against, and that the NEXT frame's
    /// phase 1 then reuses as its predictor.
    ///
    /// The MAX-reduced pyramid built here is strictly conservative and is reused
    /// by the next frame's phase 1. Terrain culling never needs full-resolution
    /// mip 0: the half-sized allocation begins at the former mip 1, and the
    /// initial copy pass proportionally MAX-reduces the full-resolution depth
    /// directly into it. This saves the full 3840x2160 R32Float allocation and
    /// write/read while preserving the conservative higher-mip chain.
    pub fn build_phase2_hzb(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
    ) -> RenderResult<()> {
        self.pyramids[1 - self.previous_index].build(device, encoder, depth, true)
    }

    pub fn phase2(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_view_proj: Mat4,
        height_bounds: (f32, f32),
        first_instance: bool,
    ) -> &CullDrawResources {
        let fresh_index = 1 - self.previous_index;
        let params = self.params(current_view_proj, height_bounds, 2, true, first_instance);
        queue.write_buffer(&self.phase2_params, 0, bytemuck::bytes_of(&params));
        clear_outputs(encoder, &self.phase2, None);
        self.dispatch(encoder, &self.phase2_bind_groups[fresh_index]);
        &self.phase2
    }

    pub fn stage_stats(&self, encoder: &mut wgpu::CommandEncoder, input: &GpuLodDrawResources) {
        encoder.copy_buffer_to_buffer(&input.output_header, 0, &self.stats_readback, 0, 16);
        encoder.copy_buffer_to_buffer(&self.phase1.header, 0, &self.stats_readback, 16, 16);
        encoder.copy_buffer_to_buffer(&self.phase2.header, 0, &self.stats_readback, 32, 16);
    }

    pub fn finish_frame(
        &mut self,
        device: &wgpu::Device,
        current_view_proj: Mat4,
    ) -> RenderResult<CullingStats> {
        let slice = self.stats_readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RenderError::render("HZB culling stats callback dropped"))?
            .map_err(|error| {
                RenderError::render(format!("HZB culling stats map failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range();
        let input = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[0..16]);
        let phase1 = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[16..32]);
        let phase2 = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[32..48]);
        drop(mapped);
        self.stats_readback.unmap();
        self.stats = CullingStats {
            frustum_passing: input.draw_count,
            phase1_drawn: phase1.draw_count,
            phase1_rejected: phase1.rejected_count,
            phase2_recovered: phase2.draw_count,
            final_drawn: phase1.draw_count + phase2.draw_count,
            culled: phase1.rejected_count.saturating_sub(phase2.draw_count),
            projection_bypassed: phase1._pad0,
            background_bypassed: phase1._pad1,
        };
        self.previous_view_proj = current_view_proj;
        self.previous_valid = true;
        self.previous_index = 1 - self.previous_index;
        Ok(self.stats)
    }

    pub fn stats(&self) -> CullingStats {
        self.stats
    }

    pub fn phase1_resources(&self) -> &CullDrawResources {
        &self.phase1
    }

    pub fn phase2_resources(&self) -> &CullDrawResources {
        &self.phase2
    }

    fn params(
        &self,
        view_proj: Mat4,
        height_bounds: (f32, f32),
        phase: u32,
        valid: bool,
        first_instance: bool,
    ) -> CullParams {
        CullParams {
            view_proj: view_proj.to_cols_array_2d(),
            height_bounds: [height_bounds.0, height_bounds.1, 0.0, 0.0],
            viewport: [
                self.width as f32,
                self.height as f32,
                self.max_mip as f32,
                u32::from(valid) as f32,
            ],
            control: [phase, u32::from(first_instance), self.mip_bias, 0],
        }
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup) {
        crate::core::shader_registry::record_shader_use("hzb_cull");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain.hzb_cull.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(self.phase1.max_draw_count.div_ceil(64), 1, 1);
    }
}

impl CullDrawResources {
    pub fn count_buffer(&self) -> &TrackedBuffer {
        &self.header
    }
}

fn create_draw_resources(
    device: &wgpu::Device,
    max_draw_count: u32,
    phase: &str,
) -> RenderResult<CullDrawResources> {
    let indirect_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&format!("terrain.hzb_cull.{phase}.indirect")),
            size: u64::from(max_draw_count) * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let instance_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&format!("terrain.hzb_cull.{phase}.instances")),
            size: u64::from(max_draw_count) * std::mem::size_of::<ClipmapDrawInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let header = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(&format!("terrain.hzb_cull.{phase}.header")),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    Ok(CullDrawResources {
        indirect_buffer,
        instance_buffer,
        max_draw_count,
        header,
    })
}

/// The `hzb_cull.wgsl` bind-group layout. Shared with the shader-differential
/// test so the test cannot drift away from the production binding contract.
fn create_cull_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain.hzb_cull.layout"),
        entries: &[
            uniform_entry(0),
            texture_entry(1),
            storage_entry(2, false),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
            storage_entry(9, false),
        ],
    })
}

/// Compile `src/shaders/hzb_cull.wgsl` into the two-phase culling pipeline.
fn create_cull_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "hzb_cull",
        HZB_CULL_SOURCE,
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain.hzb_cull.pipeline_layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    crate::core::shader_registry::create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("terrain.hzb_cull.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &TrackedBuffer,
    hzb: &wgpu::TextureView,
    input_header: &TrackedBuffer,
    input: &GpuLodDrawResources,
    output: &CullDrawResources,
    rejected: &TrackedBuffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            entry(0, params),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hzb),
            },
            entry(2, input_header),
            entry(3, &input.output_tiles),
            entry(4, &input.indirect_buffer),
            entry(5, &input.instance_buffer),
            entry(6, &output.indirect_buffer),
            entry(7, &output.instance_buffer),
            entry(8, rejected),
            entry(9, &output.header),
        ],
    })
}

fn entry(binding: u32, buffer: &TrackedBuffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
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

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn clear_outputs(
    encoder: &mut wgpu::CommandEncoder,
    output: &CullDrawResources,
    rejected: Option<&TrackedBuffer>,
) {
    encoder.clear_buffer(&output.header, 0, None);
    encoder.clear_buffer(&output.indirect_buffer, 0, None);
    encoder.clear_buffer(&output.instance_buffer, 0, None);
    if let Some(rejected) = rejected {
        encoder.clear_buffer(rejected, 0, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::resource_tracker::tracked_create_texture;
    use crate::terrain::clipmap::gpu_lod::TileInfo;
    use glam::Vec2;
    use std::collections::BTreeSet;

    /// Depth at or above which `tile_occluded` treats the covered footprint as
    /// cleared background and bypasses the test entirely.
    const BACKGROUND_DEPTH: f32 = 0.999999;
    /// Slack added to the occluder before the depth comparison.
    const DEPTH_EPSILON: f32 = 0.00001;

    /// The rule `tile_occluded` (src/shaders/hzb_cull.wgsl) applies once the
    /// covered HZB footprint has been gathered: BOTH phases use a MAX-reduced
    /// conservative pyramid and treat a cleared texel as "no occluder". Phase 1
    /// reads the previous frame under its previous view-projection; phase 2 reads
    /// the fresh phase-1 depth under the current view-projection.
    fn shader_occludes(nearest_tile_depth: f32, covered_texels: &[f32]) -> bool {
        let farthest_occluder = covered_texels.iter().copied().fold(0.0_f32, f32::max);
        if farthest_occluder >= BACKGROUND_DEPTH {
            return false;
        }
        nearest_tile_depth > farthest_occluder + DEPTH_EPSILON
    }

    fn phase2_occluded(nearest_tile_depth: f32, covered_depths: &[f32]) -> bool {
        shader_occludes(nearest_tile_depth, covered_depths)
    }

    fn relative_hzb_mip(requested_mip: u32, mip_bias: u32, max_mip: u32) -> u32 {
        requested_mip.saturating_sub(mip_bias).min(max_mip)
    }

    #[test]
    fn half_resolution_pyramid_maps_full_resolution_mips_without_unwritten_levels() {
        assert_eq!(terrain_hzb_extent(3840), 1920);
        assert_eq!(terrain_hzb_extent(2160), 1080);
        assert_eq!(terrain_hzb_extent(5), 2);
        assert_eq!(terrain_hzb_extent(1), 1);
        assert_eq!(relative_hzb_mip(0, 1, 11), 0);
        assert_eq!(relative_hzb_mip(1, 1, 11), 0);
        assert_eq!(relative_hzb_mip(2, 1, 11), 1);
        assert_eq!(relative_hzb_mip(12, 1, 11), 11);
        assert_eq!(relative_hzb_mip(0, 0, 0), 0);
    }

    #[test]
    fn fresh_hzb_recovers_previous_frame_false_negative() {
        let previous = [0.2, 0.2, 0.2, 0.2];
        let fresh = [0.2, 0.2, 1.0, 1.0];
        assert!(phase2_occluded(0.8, &previous));
        assert!(!phase2_occluded(0.8, &fresh));
    }

    #[test]
    fn max_depth_test_only_culls_when_every_covered_pixel_is_closer() {
        assert!(phase2_occluded(0.8, &[0.2, 0.4, 0.6]));
        assert!(!phase2_occluded(0.8, &[0.2, 1.0, 0.6]));
    }

    /// Read the float literal that follows `marker` in the compiled WGSL.
    fn wgsl_float_after(marker: &str) -> f32 {
        let (_, tail) = HZB_CULL_SOURCE
            .split_once(marker)
            .unwrap_or_else(|| panic!("hzb_cull.wgsl no longer contains `{marker}`"));
        let literal: String = tail
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        literal
            .parse()
            .unwrap_or_else(|_| panic!("no float literal after `{marker}`: {literal:?}"))
    }

    /// The GPU differential below is `#[ignore]`d, so this adapter-free gate is
    /// what keeps the CPU conservativeness model above honest on every host: it
    /// fails the moment the shader changes its reduce, its background cutoff or
    /// its depth slack.
    #[test]
    fn cpu_predicate_tracks_the_compiled_wgsl_source() {
        naga::front::wgsl::parse_str(HZB_CULL_SOURCE).expect("valid HZB cull WGSL");
        assert!(
            HZB_CULL_SOURCE.contains("farthest_occluder = max(farthest_occluder, depth);"),
            "hzb_cull.wgsl no longer MAX-reduces the covered footprint; the CPU model in \
             this module is stale"
        );
        assert!(HZB_CULL_SOURCE.contains("let occluder = farthest_occluder;"));
        assert!(HZB_CULL_SOURCE
            .contains("let relative_mip = requested_mip - min(requested_mip, params.control.z);"));
        assert!(HZB_CULL_SOURCE.contains("let mip = min(relative_mip, u32(params.viewport.z));"));
        assert_eq!(wgsl_float_after("if occluder >= "), BACKGROUND_DEPTH);
        assert_eq!(
            wgsl_float_after("return nearest_depth > occluder + "),
            DEPTH_EPSILON
        );
    }

    /// Synthetic occluder cases, one per cell of a `GRID`x`GRID` split of the
    /// HZB: `(nearest tile depth, footprint fill, optional single-texel
    /// override)`. Chosen to straddle the background cutoff, the depth slack and
    /// the max reduce in both directions.
    const CASES: [(f32, f32, Option<(usize, f32)>); 16] = [
        (0.80, 1.0, None),              // cleared footprint -> background bypass
        (0.80, 0.20, None),             // fully occluded
        (0.80, 0.20, Some((137, 1.0))), // one cleared texel -> conservative keep
        (0.50, 0.90, None),             // tile in front of the occluder
        (0.80, 0.20, Some((5, 0.60))),  // max reduce picks the farthest occluder
        (0.50, 0.50, None),             // exactly equal -> keep
        (0.501, 0.50, None),            // beyond the slack -> reject
        (0.80, 0.9999, None),           // just below the background cutoff
        (0.30, 0.20, None),             // shallow tile still behind
        (0.20, 0.20, None),             // equal at low depth -> keep
        (0.95, 0.90, None),             // near-background occluder still rejects
        (0.95, 0.90, Some((255, 1.0))), // ... unless one texel is cleared
        (0.10, 0.05, None),
        (0.10, 0.50, None),
        (0.60, 0.20, Some((0, 0.59))),
        (0.60, 0.20, Some((0, 0.61))),
    ];

    const GRID: u32 = 4;
    const CELL: u32 = 16;
    const HZB_DIM: u32 = GRID * CELL;

    /// The `CELL`x`CELL` HZB block covered by case `case`.
    fn footprint(case: usize) -> Vec<f32> {
        let (_, base, override_texel) = CASES[case];
        let mut texels = vec![base; (CELL * CELL) as usize];
        if let Some((index, value)) = override_texel {
            texels[index] = value;
        }
        texels
    }

    /// The whole synthetic pyramid level: cleared to 1.0, then each case's cell
    /// painted with that case's footprint.
    fn synthetic_hzb() -> Vec<f32> {
        let mut texels = vec![1.0_f32; (HZB_DIM * HZB_DIM) as usize];
        for case in 0..CASES.len() {
            let col = case as u32 % GRID;
            let row = case as u32 / GRID;
            for (local, value) in footprint(case).into_iter().enumerate() {
                let x = col * CELL + local as u32 % CELL;
                let y = row * CELL + local as u32 / CELL;
                texels[(y * HZB_DIM + x) as usize] = value;
            }
        }
        texels
    }

    /// One tile per case. With an identity view-projection world space IS clip
    /// space, so these bounds land exactly on the case's HZB cell: x maps to
    /// `u = x * 0.5 + 0.5` and y to `v = -y * 0.5 + 0.5`, and every coordinate
    /// here is a multiple of 0.5 so the mapping is exact in f32.
    fn synthetic_tiles() -> Vec<TileInfo> {
        CASES
            .iter()
            .enumerate()
            .map(|(case, (nearest, _, _))| {
                let col = case as u32 % GRID;
                let row = case as u32 / GRID;
                TileInfo::new(
                    0,
                    col,
                    row,
                    Vec2::new(col as f32 / 2.0 - 1.0, 0.5 - row as f32 / 2.0),
                    Vec2::new((col + 1) as f32 / 2.0 - 1.0, 1.0 - row as f32 / 2.0),
                )
                .with_height_bounds(*nearest, *nearest)
            })
            .collect()
    }

    fn storage_buffer<T: Pod>(
        device: &wgpu::Device,
        label: &str,
        contents: &[T],
        extra: wgpu::BufferUsages,
    ) -> TrackedBuffer {
        tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(contents),
                usage: wgpu::BufferUsages::STORAGE | extra,
            },
        )
        .expect("synthetic hzb_cull storage buffer")
    }

    /// Dispatch the REAL `hzb_cull.wgsl` phase-1 kernel over a synthetic
    /// occluder set and require its accept/reject partition to equal
    /// [`shader_occludes`]. Without this the CPU conservativeness tests above
    /// only check a hand-written copy of the predicate and would survive any
    /// change to the shader.
    #[test]
    #[ignore = "requires a physical GPU; the TESSELLA lane runs this exact test"]
    fn hzb_cull_shader_matches_the_cpu_occlusion_predicate() {
        let context = crate::core::gpu::try_ctx()
            .expect("TESSELLA HZB shader differential requires a GPU adapter");
        let device = context.device.as_ref();
        let queue = context.queue.as_ref();

        let cases = CASES.len();
        let expected_occluded: Vec<bool> = CASES
            .iter()
            .enumerate()
            .map(|(case, (nearest, _, _))| shader_occludes(*nearest, &footprint(case)))
            .collect();
        let expected_background = (0..cases)
            .filter(|case| {
                footprint(*case).iter().copied().fold(0.0_f32, f32::max) >= BACKGROUND_DEPTH
            })
            .count();
        assert!(
            expected_occluded.iter().filter(|hidden| **hidden).count() >= 4
                && expected_occluded.iter().filter(|hidden| !**hidden).count() >= 4,
            "the synthetic case table must exercise both outcomes: {expected_occluded:?}"
        );

        let hzb = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.hzb_cull.test.hzb"),
                size: wgpu::Extent3d {
                    width: HZB_DIM,
                    height: HZB_DIM,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
        .expect("synthetic HZB texture");
        let hzb_texels = synthetic_hzb();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &hzb,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(hzb_texels.as_slice()),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(HZB_DIM * 4),
                rows_per_image: Some(HZB_DIM),
            },
            wgpu::Extent3d {
                width: HZB_DIM,
                height: HZB_DIM,
                depth_or_array_layers: 1,
            },
        );
        let hzb_view = hzb.create_view(&wgpu::TextureViewDescriptor::default());

        // `viewport.z` pins the mip to 0 so the kernel reads exactly the level
        // painted above; `viewport.w` marks the pyramid valid so no tile takes
        // the projection bypass.
        let params = CullParams {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            height_bounds: [0.0, 1.0, 0.0, 0.0],
            viewport: [HZB_DIM as f32, HZB_DIM as f32, 0.0, 1.0],
            control: [1, 1, 0, 0],
        };
        let params_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.hzb_cull.test.params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )
        .expect("synthetic hzb_cull params");

        let none = wgpu::BufferUsages::empty();
        let tiles = synthetic_tiles();
        let zero_args = vec![DrawIndexedIndirectArgs::zeroed(); cases];
        let instances: Vec<ClipmapDrawInstance> = (0..cases as u32)
            .map(|index| ClipmapDrawInstance::identity(index, 0))
            .collect();
        let unrejected = vec![u32::MAX; cases];
        let input_header = storage_buffer(
            device,
            "terrain.hzb_cull.test.input_header",
            &[cases as u32, 0, 0, 0],
            none,
        );
        let input_tiles = storage_buffer(
            device,
            "terrain.hzb_cull.test.input_tiles",
            tiles.as_slice(),
            none,
        );
        let input_args = storage_buffer(
            device,
            "terrain.hzb_cull.test.input_args",
            zero_args.as_slice(),
            none,
        );
        let input_instances = storage_buffer(
            device,
            "terrain.hzb_cull.test.input_instances",
            instances.as_slice(),
            none,
        );
        let output_args = storage_buffer(
            device,
            "terrain.hzb_cull.test.output_args",
            zero_args.as_slice(),
            none,
        );
        let output_instances = storage_buffer(
            device,
            "terrain.hzb_cull.test.output_instances",
            instances.as_slice(),
            none,
        );
        let rejected = storage_buffer(
            device,
            "terrain.hzb_cull.test.rejected",
            unrejected.as_slice(),
            wgpu::BufferUsages::COPY_SRC,
        );
        let output_header = storage_buffer(
            device,
            "terrain.hzb_cull.test.output_header",
            &[0_u32; 4],
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback_size = 16 + (cases * 4) as u64;
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.hzb_cull.test.readback"),
                size: readback_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .expect("synthetic hzb_cull readback");

        let layout = create_cull_layout(device);
        let pipeline = create_cull_pipeline(device, &layout);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.hzb_cull.test.bind_group"),
            layout: &layout,
            entries: &[
                entry(0, &params_buffer),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&hzb_view),
                },
                entry(2, &input_header),
                entry(3, &input_tiles),
                entry(4, &input_args),
                entry(5, &input_instances),
                entry(6, &output_args),
                entry(7, &output_instances),
                entry(8, &rejected),
                entry(9, &output_header),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain.hzb_cull.test.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.hzb_cull.test.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((cases as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_header, 0, &readback, 0, 16);
        encoder.copy_buffer_to_buffer(&rejected, 0, &readback, 16, (cases * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .expect("hzb_cull readback callback dropped")
            .expect("hzb_cull readback map failed");
        let mapped = slice.get_mapped_range();
        let header = bytemuck::pod_read_unaligned::<CullHeader>(&mapped[0..16]);
        let rejected_ids: Vec<u32> = mapped[16..]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        drop(mapped);
        readback.unmap();

        let draw_count = header.draw_count as usize;
        let rejected_count = header.rejected_count as usize;
        assert_eq!(
            header._pad0, 0,
            "no synthetic tile may take the projection bypass; the CPU model does \
             not implement it"
        );
        assert_eq!(header._pad1 as usize, expected_background);
        assert_eq!(
            draw_count + rejected_count,
            cases,
            "every tile must be either emitted or rejected exactly once"
        );

        let gpu_occluded: BTreeSet<usize> = rejected_ids[..rejected_count]
            .iter()
            .map(|index| *index as usize)
            .collect();
        let cpu_occluded: BTreeSet<usize> = expected_occluded
            .iter()
            .enumerate()
            .filter(|(_, hidden)| **hidden)
            .map(|(case, _)| case)
            .collect();
        assert_eq!(
            gpu_occluded, cpu_occluded,
            "hzb_cull.wgsl disagrees with the CPU occlusion predicate"
        );
    }
}
