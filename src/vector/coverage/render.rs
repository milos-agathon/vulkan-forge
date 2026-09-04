use super::{
    raster::CoverageRasterPlan, BinLayout, CoverageBinner, CoverageGeometry, CoverageRasterizer,
    CoverageResolver, PrimitiveKind,
};
use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use std::sync::Arc;
use std::time::Instant;

const COVERAGE_MEMORY_BUDGET: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct CoverageRenderStats {
    pub tile_columns: u32,
    pub tile_rows: u32,
    pub tile_capacity: u32,
    pub measured_memberships: u64,
    pub written_memberships: u64,
    pub populated_tiles: usize,
    pub active_pixel_count: u32,
    pub resolve_pixel_count: u32,
    pub dispatch_retries: u32,
    pub allocation_bytes: u64,
    pub primitive_count: usize,
    pub line_count: usize,
    pub arc_count: usize,
    pub wall_ms: f64,
    pub output_sha256: String,
}

pub struct CoverageRenderOutput {
    pub coverage: Vec<f32>,
    pub linear_rgba: Vec<f32>,
    pub rgba8: Vec<u8>,
    pub errors: [u32; 4],
    pub stats: CoverageRenderStats,
}

pub(crate) struct CoverageCompiledScene {
    geometry: CoverageGeometry,
    bin_layout: BinLayout,
    raster_plan: CoverageRasterPlan,
}

impl CoverageCompiledScene {
    pub(crate) fn geometry(&self) -> &CoverageGeometry {
        &self.geometry
    }
}

pub(crate) fn compile_coverage(geometry: CoverageGeometry) -> RenderResult<CoverageCompiledScene> {
    let bin_layout = BinLayout::measure(&geometry)?;
    let raster_plan = CoverageRasterPlan::compile(&geometry)?;
    Ok(CoverageCompiledScene {
        geometry,
        bin_layout,
        raster_plan,
    })
}

/// Execute LIMES bin, analytic raster, and conflation-free resolve in order.
///
/// All three pass labels and shader hashes are recorded into the active render
/// certificate. Readbacks are included in the same 512 MiB fail-closed budget.
pub fn render_coverage(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    geometry: &CoverageGeometry,
) -> RenderResult<CoverageRenderOutput> {
    let bin_layout = BinLayout::measure(geometry)?;
    let raster_plan = CoverageRasterPlan::compile(geometry)?;
    render_prepared_coverage(device, queue, geometry, bin_layout, &raster_plan)
}

pub(crate) fn render_compiled_coverage(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    compiled: &CoverageCompiledScene,
) -> RenderResult<CoverageRenderOutput> {
    render_prepared_coverage(
        device,
        queue,
        &compiled.geometry,
        compiled.bin_layout,
        &compiled.raster_plan,
    )
}

fn render_prepared_coverage(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    geometry: &CoverageGeometry,
    bin_layout: BinLayout,
    raster_plan: &CoverageRasterPlan,
) -> RenderResult<CoverageRenderOutput> {
    let started = Instant::now();
    let binner = CoverageBinner::new(device);
    let bins = binner.prepare_compiled(device, geometry, bin_layout)?;
    let rasterizer = CoverageRasterizer::new(device);
    let raster = rasterizer.prepare_compiled(device, geometry, &bins, raster_plan)?;
    let resolver = CoverageResolver::new(device);
    let resolved = resolver.prepare(device, geometry, &bins, &raster)?;

    // `prepare` initializes primitive, baseline, rule, color, and parameter
    // buffers before this dispatch. Commit those staging writes explicitly:
    // Metal can otherwise begin a later render against zero-filled buffers
    // even though the command buffer submission is queue-ordered. This matches
    // the established vector upload boundary in `UploadedVectorScene`.
    let upload = queue.submit(std::iter::empty());
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(upload));

    let tile_count_bytes = u64::from(bins.layout.tile_count)
        .checked_mul(u64::from(bins.layout.layer_count))
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            RenderError::Budget("vector_coverage_readback_budget: tile-count overflow".into())
        })?;
    let readback_bytes = raster
        .coverage_bytes
        .checked_add(resolved.output_bytes)
        .and_then(|bytes| bytes.checked_add(16))
        .and_then(|bytes| bytes.checked_add(tile_count_bytes))
        .ok_or_else(|| {
            RenderError::Budget("vector_coverage_readback_budget: byte-count overflow".into())
        })?;
    let allocation_bytes = resolved
        .allocation_bytes
        .checked_add(readback_bytes)
        .ok_or_else(|| {
            RenderError::Budget("vector_coverage_readback_budget: total-byte overflow".into())
        })?;
    if allocation_bytes > COVERAGE_MEMORY_BUDGET {
        return Err(RenderError::Budget(format!(
            "vector_coverage_readback_budget: required={allocation_bytes} \
             budget={COVERAGE_MEMORY_BUDGET}; refusing to truncate"
        )));
    }

    let coverage_readback =
        readback_buffer(device, "vf.Vector.Coverage.Readback", raster.coverage_bytes)?;
    let resolved_readback = readback_buffer(
        device,
        "vf.Vector.Coverage.ResolvedReadback",
        resolved.output_bytes,
    )?;
    let error_readback = readback_buffer(device, "vf.Vector.Coverage.ErrorReadback", 16)?;
    let tile_count_readback = readback_buffer(
        device,
        "vf.Vector.Coverage.TileCountReadback",
        tile_count_bytes.max(4),
    )?;

    let mut timing =
        crate::core::gpu_timing::OneShotTiming::for_device(device.clone(), queue.clone());
    let mut bin_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.Coverage.BinEncoder"),
    });
    let bin_scope = timing.begin(&mut bin_encoder, "vector.coverage.bin");
    binner.encode(
        &mut bin_encoder,
        &bins,
        u32::try_from(geometry.primitives.len()).map_err(|_| {
            RenderError::Budget("vector_coverage_render: primitive count exceeds u32".into())
        })?,
    );
    timing.end(&mut bin_encoder, bin_scope, 1);
    let bin_submission = queue.submit(Some(bin_encoder.finish()));
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(bin_submission));
    let mut tile_counts =
        read_bin_counts(device, queue, &bins, &tile_count_readback, tile_count_bytes)?;
    let mut written_memberships = tile_counts.iter().map(|&count| u64::from(count)).sum();
    let mut retry_count = 0_u32;
    while written_memberships != bins.layout.measured_memberships && retry_count < 3 {
        retry_count += 1;
        let mut retry_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Coverage.BinRetryEncoder"),
        });
        binner.encode(
            &mut retry_encoder,
            &bins,
            u32::try_from(geometry.primitives.len()).map_err(|_| {
                RenderError::Budget("vector_coverage_render: primitive count exceeds u32".into())
            })?,
        );
        let retry_submission = queue.submit(Some(retry_encoder.finish()));
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(retry_submission));
        tile_counts =
            read_bin_counts(device, queue, &bins, &tile_count_readback, tile_count_bytes)?;
        written_memberships = tile_counts.iter().map(|&count| u64::from(count)).sum();
    }
    if written_memberships != bins.layout.measured_memberships {
        return Err(RenderError::Render(format!(
            "{{\"status\":\"vector_coverage_bin_dispatch_mismatch\",\
             \"measured_memberships\":{},\"written_memberships\":{},\"attempts\":{}}}",
            bins.layout.measured_memberships,
            written_memberships,
            retry_count + 1
        )));
    }
    if retry_count != 0 {
        crate::core::degradation::record_degradation(
            "dispatch_retry",
            "vector.coverage.bin",
            &format!(
                "adapter required {retry_count} verified retry submission(s) before the measured \
                 membership count was complete"
            ),
        );
    }

    let mut raster_retry_count = 0_u32;
    loop {
        let mut raster_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Coverage.RasterEncoder"),
        });
        let raster_scope = (raster_retry_count == 0)
            .then(|| timing.begin(&mut raster_encoder, "vector.coverage.raster"))
            .flatten();
        rasterizer.encode(&mut raster_encoder, &raster, geometry);
        if raster_retry_count == 0 {
            timing.end(&mut raster_encoder, raster_scope, 1);
        }
        let raster_submission = queue.submit(Some(raster_encoder.finish()));
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(raster_submission));
        let stage_words = read_error_words(device, queue, &bins, &error_readback)?;
        if raster.active_pixel_count == 0 || stage_words[3] & 1 != 0 {
            break;
        }
        raster_retry_count += 1;
        if raster_retry_count >= 3 {
            return Err(RenderError::Render(
                "{\"status\":\"vector_coverage_raster_dispatch_missing\",\"attempts\":3}".into(),
            ));
        }
    }

    let mut resolve_retry_count = 0_u32;
    loop {
        let mut resolve_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Coverage.ResolveEncoder"),
        });
        let resolve_scope = (resolve_retry_count == 0)
            .then(|| timing.begin(&mut resolve_encoder, "vector.coverage.resolve"))
            .flatten();
        resolver.encode(&mut resolve_encoder, &resolved);
        if resolve_retry_count == 0 {
            timing.end(&mut resolve_encoder, resolve_scope, 1);
        }
        let resolve_submission = queue.submit(Some(resolve_encoder.finish()));
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(resolve_submission));
        let stage_words = read_error_words(device, queue, &bins, &error_readback)?;
        if resolved.active_pixel_count == 0 || stage_words[3] & 2 != 0 {
            break;
        }
        resolve_retry_count += 1;
        if resolve_retry_count >= 3 {
            return Err(RenderError::Render(
                "{\"status\":\"vector_coverage_resolve_dispatch_missing\",\"attempts\":3}".into(),
            ));
        }
    }
    if raster_retry_count != 0 {
        crate::core::degradation::record_degradation(
            "dispatch_retry",
            "vector.coverage.raster",
            &format!(
                "adapter required {raster_retry_count} verified retry submission(s) before the \
                 raster stage marker was observed"
            ),
        );
    }
    if resolve_retry_count != 0 {
        crate::core::degradation::record_degradation(
            "dispatch_retry",
            "vector.coverage.resolve",
            &format!(
                "adapter required {resolve_retry_count} verified retry submission(s) before the \
                 resolve stage marker was observed"
            ),
        );
    }

    let mut readback_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.Coverage.ReadbackEncoder"),
    });
    timing.resolve(&mut readback_encoder);
    readback_encoder.copy_buffer_to_buffer(
        &raster.coverage,
        0,
        &coverage_readback,
        0,
        raster.coverage_bytes,
    );
    readback_encoder.copy_buffer_to_buffer(
        &resolved.output,
        0,
        &resolved_readback,
        0,
        resolved.output_bytes,
    );
    readback_encoder.copy_buffer_to_buffer(&bins.overflow, 0, &error_readback, 0, 16);
    let readback_submission = queue.submit(Some(readback_encoder.finish()));
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(readback_submission));

    if !timing.record_into_certificate() {
        crate::core::certificate::record_pass("vector.coverage.bin", 0.0, 1);
        crate::core::certificate::record_pass("vector.coverage.raster", 0.0, 1);
        crate::core::certificate::record_pass("vector.coverage.resolve", 0.0, 1);
    }

    let error_words = map_buffer::<u32>(device, &error_readback)?;
    let errors: [u32; 4] = error_words
        .as_slice()
        .try_into()
        .map_err(|_| RenderError::Readback("LIMES error block is not four words".into()))?;
    if errors[..3] != [0; 3] {
        return Err(RenderError::Render(format!(
            "{{\"status\":\"vector_coverage_overflow\",\"bin\":{},\
             \"active_list\":{},\"breakpoints\":{},\"dispatch_stage_bits\":{}}}",
            errors[0], errors[1], errors[2], errors[3]
        )));
    }

    let structured_errors = [errors[0], errors[1], errors[2], 0];
    let coverage = map_buffer::<f32>(device, &coverage_readback)?;
    let linear_rgba = map_buffer::<f32>(device, &resolved_readback)?;
    let rgba8 = linear_to_straight_rgba8(&linear_rgba);
    let output_sha256 = crate::core::provenance::to_hex(&crate::core::provenance::sha256(&rgba8));
    let line_count = geometry
        .primitives
        .iter()
        .filter(|record| record.kind() == PrimitiveKind::Line)
        .count();
    let primitive_count = geometry.primitives.len();
    Ok(CoverageRenderOutput {
        coverage,
        linear_rgba,
        rgba8,
        errors: structured_errors,
        stats: CoverageRenderStats {
            tile_columns: bins.layout.tile_columns,
            tile_rows: bins.layout.tile_rows,
            tile_capacity: bins.layout.tile_capacity,
            measured_memberships: bins.layout.measured_memberships,
            written_memberships,
            populated_tiles: tile_counts.iter().filter(|&&count| count != 0).count(),
            active_pixel_count: raster.active_pixel_count,
            resolve_pixel_count: raster.resolve_pixel_count,
            dispatch_retries: retry_count + raster_retry_count + resolve_retry_count,
            allocation_bytes,
            primitive_count,
            line_count,
            arc_count: primitive_count - line_count,
            wall_ms: started.elapsed().as_secs_f64() * 1_000.0,
            output_sha256,
        },
    })
}

fn read_bin_counts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bins: &super::CoverageBins,
    readback: &wgpu::Buffer,
    byte_count: u64,
) -> RenderResult<Vec<u32>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.Coverage.BinVerifyEncoder"),
    });
    encoder.copy_buffer_to_buffer(&bins.tile_counts, 0, readback, 0, byte_count);
    let submission = queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
    map_buffer::<u32>(device, readback)
}

fn read_error_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bins: &super::CoverageBins,
    readback: &wgpu::Buffer,
) -> RenderResult<[u32; 4]> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.Coverage.StageVerifyEncoder"),
    });
    encoder.copy_buffer_to_buffer(&bins.overflow, 0, readback, 0, 16);
    let submission = queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
    map_buffer::<u32>(device, readback)?
        .try_into()
        .map_err(|_| RenderError::Readback("LIMES stage block is not four words".into()))
}

fn readback_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
) -> RenderResult<TrackedBuffer> {
    tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )
}

fn map_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> RenderResult<Vec<T>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver.receive())
        .ok_or_else(|| RenderError::Readback("LIMES map callback was cancelled".into()))?
        .map_err(|error| RenderError::Readback(format!("LIMES map failed: {error}")))?;
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(result)
}

fn linear_to_straight_rgba8(linear: &[f32]) -> Vec<u8> {
    // Writes through a pre-sized buffer instead of pushing one byte at a time.
    // At 1920x1080 the push form spent 18.7 ms on 8.3M capacity-checked pushes
    // against 14.6 ms here. The per-channel arithmetic and its order are
    // unchanged, so the RGBA8 output and its provenance hash are identical.
    let mut output = vec![0_u8; (linear.len() / 4) * 4];
    for (bytes, pixel) in output.chunks_exact_mut(4).zip(linear.chunks_exact(4)) {
        let alpha = pixel[3].clamp(0.0, 1.0);
        let inverse_alpha = if alpha > 0.0 { alpha.recip() } else { 0.0 };
        for channel in 0..3 {
            let straight = (pixel[channel] * inverse_alpha).clamp(0.0, 1.0);
            bytes[channel] = (straight * 255.0).round() as u8;
        }
        bytes[3] = (alpha * 255.0).round() as u8;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::linear_to_straight_rgba8;

    #[test]
    fn resolved_linear_premultiplied_values_convert_to_straight_rgba8() {
        assert_eq!(
            linear_to_straight_rgba8(&[0.25, 0.0, 0.5, 0.75]),
            [85, 0, 170, 191]
        );
        assert_eq!(linear_to_straight_rgba8(&[0.0; 4]), [0, 0, 0, 0]);
    }
}
