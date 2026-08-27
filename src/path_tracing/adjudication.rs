// src/path_tracing/adjudication.rs
// AEQUITAS path-traced ground truth: drives the existing WavefrontScheduler
// over the analytic ReferenceSceneDesc at high accumulated spp. This is a
// genuine multi-bounce wavefront path trace (NEE at every hit, BSDF-sampled
// continuation rays up to the scheduler depth cap, Russian roulette from
// depth 4) — every frame must execute at least two wavefront iterations
// (primary wave + a bounce wave) or rendering fails.
// ReSTIR is fully disabled for determinism (no temporal history leaks into the
// single-frame reference); the scene-spatial bind group is still initialized
// and asserted Some so the ReSTIR spatial stage would be fully scene-bound if
// enabled (dispatch_restir_spatial cannot early-return for a missing scene).
// RELEVANT FILES: src/path_tracing/wavefront/render.rs, src/path_tracing/reference_scene.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, TrackedBuffer,
};
use crate::path_tracing::reference_scene::ReferenceSceneDesc;
use crate::path_tracing::wavefront::WavefrontScheduler;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{Device, Queue};

/// Uniforms layout shared by every wavefront kernel (pt_raygen.wgsl et al.).
/// 96 bytes; identical field placement to `compute_types::Uniforms`, but the
/// 4th u32 is `spp` in the wavefront kernels (aov_flags in the megakernel).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WavefrontUniforms {
    width: u32,
    height: u32,
    frame_index: u32,
    spp: u32,
    cam_origin: [f32; 3],
    cam_fov_y: f32,
    cam_right: [f32; 3],
    cam_aspect: f32,
    cam_up: [f32; 3],
    cam_exposure: f32,
    cam_forward: [f32; 3],
    seed_hi: u32,
    seed_lo: u32,
    _pad_end: [u32; 3],
}

fn storage_buffer(
    device: &Device,
    label: &str,
    contents: &[u8],
) -> Result<TrackedBuffer, RenderError> {
    tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )
}

/// Render the linear-HDR path-traced reference: mean radiance over
/// `spp_frames` accumulated frames of the full wavefront path tracer
/// (multi-bounce: NEE at every hit, BSDF-sampled continuation rays up to the
/// scheduler's depth cap, Russian roulette from depth 4). One camera sample
/// per pixel per frame; the per-pixel mean over `spp_frames` converges to the
/// full light-transport integral of the reference scene, not a direct-lighting
/// proxy. Each frame must execute at least two wavefront iterations
/// (primary wave + at least one bounce wave) or rendering fails.
/// Returns RGBA f32, row-major, `width * height * 4` values, alpha = 1.
///
/// `timing` (CENSOR F-04): when supplied, the first frame's primary wavefront
/// wave is timed under the "adjudication.path_trace" certificate label (see
/// `render_frame_simple` — a whole frame cannot be bracketed on one encoder).
/// The `OneShotTiming` MUST live on the same wgpu device as `device`; pass
/// `None` when driving a standalone device (tests).
pub fn render_pt_reference(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    desc: &ReferenceSceneDesc,
    width: u32,
    height: u32,
    spp_frames: u32,
    mut timing: Option<&mut crate::core::gpu_timing::OneShotTiming>,
) -> Result<Vec<f32>, RenderError> {
    if width == 0 || height == 0 || spp_frames == 0 {
        return Err(RenderError::Render(
            "adjudication PT reference requires non-zero width/height/spp".into(),
        ));
    }

    let mut scheduler = WavefrontScheduler::new(device.clone(), queue.clone(), width, height)
        .map_err(|e| RenderError::Render(format!("wavefront scheduler init: {e}")))?;
    // Deterministic reference: no ReSTIR temporal/spatial resampling.
    scheduler.set_restir_enabled(false);
    scheduler.set_restir_spatial_enabled(false);
    scheduler.set_environment_params(&desc.environment_raw());

    // --- Scene buffers from the single ReferenceSceneDesc ---
    let spheres = desc.wavefront_spheres();
    let spheres_buffer = storage_buffer(
        device,
        "adjudication-spheres",
        bytemuck::cast_slice(&spheres),
    )?;
    let area_lights = desc.area_lights();
    let area_lights_buffer = storage_buffer(
        device,
        "adjudication-area-lights",
        bytemuck::cast_slice(&area_lights),
    )?;
    let dir_lights = desc.directional_lights();
    let dir_lights_buffer = storage_buffer(
        device,
        "adjudication-dir-lights",
        bytemuck::cast_slice(&dir_lights),
    )?;
    let importance = desc.object_importance();
    let importance_buffer = storage_buffer(
        device,
        "adjudication-importance",
        bytemuck::cast_slice(&importance),
    )?;

    // Ground plane as a real mesh BLAS so pt_intersect/pt_shadow see the exact
    // same two triangles the raster twin draws.
    let plane = desc.plane_mesh();
    let bvh = crate::accel::cpu_bvh::build_bvh_cpu(&plane, &Default::default())
        .map_err(|e| RenderError::Render(format!("plane BVH build: {e}")))?;
    let atlas_items = [(plane, bvh)];
    let atlas = crate::path_tracing::mesh::build_mesh_atlas(device, &atlas_items)
        .map_err(|e| RenderError::Render(format!("plane mesh atlas: {e}")))?;

    let ident: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let plane_instance = crate::accel::instancing::InstanceData {
        transform: ident,
        inv_transform: ident,
        blas_index: 0,
        material_id: 3, // plane material slot in the spheres array
        _padding: [0; 2],
    };
    scheduler.set_instances_buffer(storage_buffer(
        device,
        "adjudication-instances",
        bytemuck::cast_slice(&[plane_instance]),
    )?);
    scheduler.set_blas_descs_buffer(atlas.descs_buffer);

    // Fully bind the ReSTIR scene-spatial group and confirm it is Some, per
    // the adjudication contract (spatial resampling must never early-return
    // for a missing scene binding).
    scheduler
        .init_restir_scene_spatial_bind_group(&area_lights_buffer, &dir_lights_buffer)
        .map_err(|e| RenderError::Render(format!("restir scene-spatial bind group: {e}")))?;
    debug_assert!(scheduler.restir_scene_bound());
    if !scheduler.restir_scene_bound() {
        return Err(RenderError::Render(
            "restir scene-spatial bind group not bound".into(),
        ));
    }

    let scene_bind_group = scheduler
        .create_scene_bind_group(
            &spheres_buffer,
            &atlas.vertex_buffer,
            &atlas.index_buffer,
            &atlas.bvh_buffer,
            &area_lights_buffer,
            &dir_lights_buffer,
            &importance_buffer,
        )
        .map_err(|e| RenderError::Render(format!("scene bind group: {e}")))?;

    // --- Accumulation target (vec4<f32> per pixel, zero-initialized) ---
    let px_count = (width as usize) * (height as usize);
    let accum_bytes = (px_count * std::mem::size_of::<[f32; 4]>()) as u64;
    let accum_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("adjudication-accum-hdr"),
            size: accum_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let accum_bind_group = scheduler.create_accum_bind_group(&accum_buffer);

    // --- Camera uniforms exactly as pt_raygen.wgsl consumes them ---
    let (origin, forward, right, up) = desc.camera_basis();
    let mut uniforms = WavefrontUniforms {
        width,
        height,
        frame_index: 0,
        spp: 1,
        cam_origin: origin.into(),
        cam_fov_y: desc.fov_y_rad(),
        cam_right: right.into(),
        cam_aspect: width as f32 / height as f32,
        cam_up: up.into(),
        cam_exposure: desc.exposure,
        cam_forward: forward.into(),
        seed_hi: desc.seed_hi,
        seed_lo: desc.seed_lo,
        _pad_end: [1, 0, 0],
    };
    let uniforms_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("adjudication-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // --- Accumulate frames ---
    // Per-frame seed decorrelation. pt_raygen derives
    // rng_hi = seed_hi ^ (pixel * 9781) ^ (frame * 6271) and pt_shade
    // re-hashes with distinct odd constants (26699/30977), so samples
    // decorrelate per pixel AND per frame. Hashing fresh per-frame seeds here
    // (splitmix32-style) additionally decorrelates the raygen jitter and
    // Russian-roulette streams across frames, so the accumulated per-pixel
    // mean converges to the scene's full light-transport integral.
    fn splitmix32(mut x: u32) -> u32 {
        x = x.wrapping_add(0x9E37_79B9);
        let mut z = x;
        z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
        z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
        z ^ (z >> 15)
    }
    for frame in 0..spp_frames {
        uniforms.frame_index = frame;
        uniforms.seed_hi = splitmix32(desc.seed_hi ^ frame);
        uniforms.seed_lo = splitmix32(desc.seed_lo ^ frame.wrapping_mul(0x0000_9E3D));
        queue.write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
        // Only frame 0 carries the timing scope so the certificate records the
        // "adjudication.path_trace" pass exactly once.
        let frame_timing = if frame == 0 {
            timing
                .as_mut()
                .map(|t| (&mut **t, "adjudication.path_trace", spp_frames))
        } else {
            None
        };
        let iterations = scheduler
            .render_frame_simple(
                &uniforms_buffer,
                &scene_bind_group,
                &accum_bind_group,
                frame_timing,
            )
            .map_err(|e| RenderError::Render(format!("wavefront frame {frame}: {e}")))?;
        // Multi-bounce contract: a path-traced reference frame of this scene
        // always consumes the primary wave AND at least one bounce wave. One
        // iteration means the tracer degenerated into a primary-only
        // (direct-lighting) shortcut.
        if iterations < 2 {
            return Err(RenderError::Render(format!(
                "adjudication PT frame {frame} executed {iterations} wavefront iteration(s); \
                 a multi-bounce path-traced reference requires >= 2"
            )));
        }
        if frame % 64 == 63 {
            device.poll(wgpu::Maintain::Wait);
        }
    }

    // --- Readback through a tracked host-visible staging buffer ---
    // `tracked_create_buffer` records the host-visible allocation in the global
    // ledger and releases it when `staging` is dropped below.
    let staging = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("adjudication-accum-readback"),
            size: accum_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("adjudication-accum-copy"),
    });
    encoder.copy_buffer_to_buffer(&accum_buffer, 0, &staging, 0, accum_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let hdr_sum: Vec<f32> = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    };
    staging.unmap();
    drop(staging);

    // Mean over frames; force alpha to 1 (the accum alpha channel is a ReSTIR
    // diagnostic, not coverage).
    let inv = 1.0 / spp_frames as f32;
    let mut hdr = hdr_sum;
    for px in hdr.chunks_exact_mut(4) {
        px[0] *= inv;
        px[1] *= inv;
        px[2] *= inv;
        px[3] = 1.0;
    }
    Ok(hdr)
}

#[cfg(test)]
mod tests {
    #[test]
    fn adjudication_shadow_slots_have_one_accumulator_writer_per_pixel() {
        let shade = include_str!("../shaders/pt_shade.wgsl");
        let shadow = include_str!("../shaders/pt_shadow.wgsl");
        for source in [shade, shadow] {
            let module = naga::front::wgsl::parse_str(source).expect("valid wavefront WGSL");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("valid wavefront module");
        }
        for kind in 0..4 {
            assert!(shade.contains(&format!("push_shadow(sr, {kind}u)")));
        }
        assert!(shade.contains("kind * pixel_count + sr.pixel"));
        assert!(shade.contains("sr._pad1.x = 1u"));
        assert_eq!(
            shade
                .matches("atomicAdd(&shadow_queue_header.in_count")
                .count(),
            1
        );
        assert!(shadow.contains("if (pixel >= pixel_count) { return; }"));
        assert!(shadow.contains("for (var kind = 0u; kind < 4u; kind = kind + 1u)"));
        assert!(shadow.contains("kind * pixel_count + pixel"));
        assert!(shadow.contains("cleared._pad1 = vec3<u32>(0u)"));
        assert_eq!(
            shadow
                .matches("accum_hdr[pixel] = accum_hdr[pixel] +")
                .count(),
            1
        );
        let normalized_shadow = shadow.replace("\r\n", "\n");
        assert!(
            normalized_shadow
                .find("return;\n    }\n\n    // Persistent threads")
                .unwrap()
                < normalized_shadow
                    .find("atomicAdd(&shadow_queue_header.out_count")
                    .unwrap()
        );
        assert_eq!(
            std::mem::offset_of!(super::WavefrontUniforms, _pad_end),
            84,
            "adjudication_mode must map to Uniforms byte 84"
        );
        assert_eq!(std::mem::size_of::<super::WavefrontUniforms>(), 96);
        for source in [shade, shadow] {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, uniforms) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some("Uniforms"))
                .unwrap();
            let naga::TypeInner::Struct { members, .. } = &uniforms.inner else {
                panic!("Uniforms struct");
            };
            assert!(members.iter().any(|member| {
                member.name.as_deref() == Some("adjudication_mode") && member.offset == 84
            }));
        }
        assert!(include_str!("hybrid_compute/mod.rs").contains("_pad_end: [0, 0, 0]"));
        assert!(include_str!("../py_functions/path_tracing/gpu.rs").contains("_pad_end: [0, 0, 0]"));
    }

    #[test]
    fn adjudication_driver_uses_wavefront_active_count_loop_not_zero_stub() {
        let driver = include_str!("adjudication.rs");
        let render = include_str!("wavefront/render.rs");
        let queues = include_str!("wavefront/queues/types.rs");

        assert!(driver.contains(".render_frame_simple("));
        assert!(render.contains("get_active_ray_count"));
        assert!(!queues.contains("Ok(0)"));
        // Gap 2: the one-iteration escape hatch must stay dead.
        assert!(
            !render.contains("did_any"),
            "render_frame_simple must not reintroduce the did_any one-iteration fallback"
        );
        // Gap 1: the driver must enforce the multi-bounce contract per frame.
        assert!(
            driver.contains("iterations < 2"),
            "render_pt_reference must reject frames that ran fewer than two wavefront iterations"
        );
    }

    #[test]
    fn pt_reference_renders_multibounce_frames() {
        // Fails if the adjudication PT path is ever rewired as a
        // one-iteration/direct-lighting shortcut: render_pt_reference errors
        // when any frame executes fewer than two wavefront iterations.
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            return; // no GPU on this host: same silent-skip convention as queues/types.rs
        };
        let Ok((device, queue)) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("adjudication-multibounce-test"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        )) else {
            return;
        };
        let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));
        let desc = crate::path_tracing::reference_scene::adjudication_scene();
        // timing = None: this test drives a standalone device, and a timing
        // manager from the global context would live on a different device.
        let hdr = super::render_pt_reference(&device, &queue, &desc, 32, 32, 2, None)
            .expect("multi-bounce PT reference render");
        assert_eq!(hdr.len(), 32 * 32 * 4);
        assert!(hdr.iter().any(|&v| v > 0.0), "PT reference is all black");
    }
}
