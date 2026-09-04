// src/shadows/mod.rs
// Shadow mapping implementations for Workstream B
// Exists to centralize GPU/CPU shadow utilities shared across bindings and pipelines
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

mod cascade_math;
mod csm_depth_control;
mod csm_renderer;
mod csm_types;

pub mod blur_pass;
pub mod manager;
pub mod moment_pass;
pub mod state;

// Re-export CSM types from split modules
pub use cascade_math::detect_peter_panning;
pub use csm_renderer::CsmRenderer;
pub use csm_types::{CascadeStatistics, CsmConfig, CsmUniforms, ShadowCascade};

pub use blur_pass::{ShadowBlurPass, DEFAULT_MOMENT_BLUR_RADIUS};
pub use manager::{ShadowManager, ShadowManagerConfig};
pub(crate) use manager::{
    DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS, DEFAULT_PCSS_FILTER_RADIUS_TEXELS, DEFAULT_PCSS_LIGHT_SIZE,
};
pub use moment_pass::{create_moment_storage_view, MomentGenerationPass};

// Re-export common shadow types and utilities
pub use csm_renderer::CsmRenderer as CascadedShadowMaps;

pub(crate) const CSM_SHADER_SOURCE: &str = concat!(
    include_str!("../shaders/includes/determinism.wgsl"),
    "\n",
    include_str!("../shaders/includes/shadow_moments.wgsl"),
    "\n",
    include_str!("../shaders/shadows.wgsl")
);

/// Largest EVSM exponent an `Rgba16Float` moment atlas can carry.
///
/// Keeps `exp(2c)` below the largest finite binary16 value (65504).
pub const EVSM_MAX_EXPONENT_RGBA16F: f32 = 5.5;
pub(crate) const MIN_SHADOW_MAP_SIZE: u32 = 1;
pub(crate) const MAX_SHADOW_CASCADES: u32 = 4;
pub(crate) const MAX_SHADOW_ALLOCATION_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) fn validate_shadow_dimensions(
    resolution: u32,
    cascades: u32,
) -> crate::core::error::RenderResult<()> {
    if resolution < MIN_SHADOW_MAP_SIZE || !resolution.is_power_of_two() {
        return Err(crate::core::error::RenderError::render(format!(
            "shadow resolution must be a positive power of two, got {resolution}"
        )));
    }
    if !(1..=MAX_SHADOW_CASCADES).contains(&cascades) {
        return Err(crate::core::error::RenderError::render(format!(
            "shadow cascade count must be within 1..={MAX_SHADOW_CASCADES}, got {cascades}"
        )));
    }
    Ok(())
}

pub(crate) fn shadow_allocation_bytes(
    resolution: u32,
    cascades: u32,
    requires_moments: bool,
) -> crate::core::error::RenderResult<u64> {
    let bytes_per_texel = if requires_moments { 20_u64 } else { 4_u64 };
    u64::from(resolution)
        .checked_mul(u64::from(resolution))
        .and_then(|pixels| pixels.checked_mul(u64::from(cascades)))
        .and_then(|pixels| pixels.checked_mul(bytes_per_texel))
        .ok_or_else(|| crate::core::error::RenderError::render("shadow allocation size overflow"))
}

pub(crate) fn validate_shadow_allocation_budget(
    resolution: u32,
    cascades: u32,
    requires_moments: bool,
) -> crate::core::error::RenderResult<()> {
    validate_shadow_dimensions(resolution, cascades)?;
    let allocation_bytes = shadow_allocation_bytes(resolution, cascades, requires_moments)?;
    if allocation_bytes > MAX_SHADOW_ALLOCATION_BYTES {
        return Err(crate::core::error::RenderError::budget(format!(
            "shadow resources require {:.1} MiB, exceeding the 512 MiB shadow budget",
            allocation_bytes as f64 / (1024.0 * 1024.0)
        )));
    }
    Ok(())
}

pub(crate) fn validate_shadow_device_limits(
    device: &wgpu::Device,
    resolution: u32,
    cascades: u32,
) -> crate::core::error::RenderResult<()> {
    validate_shadow_dimensions(resolution, cascades)?;
    let limits = device.limits();
    if resolution > limits.max_texture_dimension_2d {
        return Err(crate::core::error::RenderError::device(format!(
            "shadow resolution {resolution} exceeds device max_texture_dimension_2d {}",
            limits.max_texture_dimension_2d
        )));
    }
    if cascades > limits.max_texture_array_layers {
        return Err(crate::core::error::RenderError::device(format!(
            "shadow cascade count {cascades} exceeds device max_texture_array_layers {}",
            limits.max_texture_array_layers
        )));
    }
    Ok(())
}

/// Clamp an EVSM exponent to the range the moment atlas can actually represent.
///
/// Must be applied identically to the moment-generation pass and to the shader
/// uniforms that sample it: producer and consumer have to warp by the same constant.
pub fn clamp_evsm_exponent(exponent: f32) -> f32 {
    if exponent.is_finite() {
        exponent.clamp(0.0, EVSM_MAX_EXPONENT_RGBA16F)
    } else {
        0.0
    }
}

pub(crate) fn requires_moment_blur(technique: crate::lighting::types::ShadowTechnique) -> bool {
    matches!(
        technique,
        crate::lighting::types::ShadowTechnique::VSM
            | crate::lighting::types::ShadowTechnique::EVSM
            | crate::lighting::types::ShadowTechnique::MSM
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_dimensions_reject_invalid_requests_before_allocation() {
        for resolution in [0, 3, 511, 513, u32::MAX] {
            assert!(validate_shadow_dimensions(resolution, 1).is_err());
        }
        for cascades in [0, 5, u32::MAX] {
            assert!(validate_shadow_dimensions(128, cascades).is_err());
        }
        assert!(validate_shadow_dimensions(1, 1).is_ok());
        assert!(validate_shadow_dimensions(128, 1).is_ok());
        assert!(validate_shadow_dimensions(512, 1).is_ok());
        assert!(validate_shadow_dimensions(16_384, 4).is_ok());
    }

    #[test]
    fn moment_shadow_allocation_counts_depth_atlas_and_intermediate() {
        assert_eq!(
            shadow_allocation_bytes(4096, 2, true).expect("valid allocation"),
            640 * 1024 * 1024
        );
    }

    #[test]
    fn viewer_shadow_budget_is_technique_aware_without_gpu_allocation() {
        let error = validate_shadow_allocation_budget(4096, 4, true)
            .expect_err("4096x4096 four-cascade moment shadows exceed the fixed budget");
        assert_eq!(
            error.to_string(),
            "Memory budget exceeded: shadow resources require 1280.0 MiB, exceeding the 512 MiB shadow budget"
        );
        assert!(
            validate_shadow_allocation_budget(4096, 4, false).is_ok(),
            "PCF/PCSS only allocate the 256 MiB depth cascade array"
        );
        assert!(validate_shadow_allocation_budget(2048, 4, true).is_ok());
    }

    fn execute_shader_probe(source: &str, label: &str, probe_body: &str) -> Option<[f32; 5]> {
        let source = format!(
            "{source}
@group(0) @binding(31) var<storage, read_write> visibility_output: array<f32, 5>;

@compute @workgroup_size(1)
fn test_visibility_entry() {{
    {probe_body}
}}"
        );
        crate::shader_sources::assert_valid_wgsl(&source);
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            eprintln!("{label}: live GPU unavailable; validated the WGSL contract statically");
            return None;
        };
        let device = &device;
        let queue = &queue;
        let module =
            crate::core::shader_registry::create_labeled_shader_module(device, label, &source);
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: "test_visibility_entry",
            },
        );
        let output = crate::core::resource_tracker::tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("evsm-visibility-output"),
                size: 20,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .expect("output buffer");
        let readback = crate::core::resource_tracker::tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("evsm-visibility-readback"),
                size: 20,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .expect("readback buffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("evsm-visibility-contract"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 31,
                resource: output.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evsm-visibility-contract"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("evsm-visibility-contract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 20);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("map callback receiver");
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let result = *bytemuck::from_bytes::<[f32; 5]>(&mapped);
        drop(mapped);
        readback.unmap();
        Some(result)
    }

    fn execute_evsm_half_uniform_probe(depths: &[f32]) -> Option<Vec<f32>> {
        let source = format!(
            "{}
@group(0) @binding(30) var<storage, read> evsm_inputs: array<vec4<f32>>;
@group(0) @binding(31) var<storage, read_write> evsm_outputs: array<f32>;

@compute @workgroup_size(64)
fn test_evsm_half_uniform(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= {}u) {{
        return;
    }}
    let input = evsm_inputs[id.x];
    evsm_outputs[id.x] =
        evsm_moment_leak_control(input.xy, input.z, 5.5, input.w);
}}",
            include_str!("../shaders/includes/shadow_moments.wgsl"),
            depths.len()
        );
        crate::shader_sources::assert_valid_wgsl(&source);
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            eprintln!(
                "EVSM half-uniform probe: live GPU unavailable; validated the WGSL contract statically"
            );
            return None;
        };
        let device = &device;
        let queue = &queue;
        let module = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "evsm-half-uniform-contract",
            &source,
        );
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("evsm-half-uniform-contract"),
                layout: None,
                module: &module,
                entry_point: "test_evsm_half_uniform",
            },
        );

        let inputs = depths
            .iter()
            .map(|&depth| {
                let receiver = (EVSM_MAX_EXPONENT_RGBA16F * (depth - 1.0)).exp();
                let stored_mean = half::f16::from_f32(receiver).to_f32();
                let stored_mean_squared = half::f16::from_f32(receiver * receiver).to_f32();
                let minimum_variance = (0.000375 * EVSM_MAX_EXPONENT_RGBA16F * receiver).powi(2);
                [stored_mean, stored_mean_squared, receiver, minimum_variance]
            })
            .collect::<Vec<_>>();
        let input = crate::core::resource_tracker::tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("evsm-half-uniform-input"),
                contents: bytemuck::cast_slice(&inputs),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )
        .expect("input buffer");
        let byte_size = (depths.len() * std::mem::size_of::<f32>()) as u64;
        let output = crate::core::resource_tracker::tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("evsm-half-uniform-output"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .expect("output buffer");
        let readback = crate::core::resource_tracker::tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("evsm-half-uniform-readback"),
                size: byte_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .expect("readback buffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("evsm-half-uniform-contract"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 31,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evsm-half-uniform-contract"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("evsm-half-uniform-contract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((depths.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, byte_size);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("map callback receiver");
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        Some(result)
    }

    #[test]
    fn evsm_exponent_clamp_keeps_normalized_moments_representable_in_rgba16f() {
        assert_eq!(clamp_evsm_exponent(-1.0), 0.0);
        assert_eq!(clamp_evsm_exponent(40.0), EVSM_MAX_EXPONENT_RGBA16F);
        assert_eq!(clamp_evsm_exponent(f32::NAN), 0.0);
        assert_eq!(clamp_evsm_exponent(f32::INFINITY), 0.0);

        for depth in [0.0_f32, 0.5, 1.0] {
            let positive = (EVSM_MAX_EXPONENT_RGBA16F * (depth - 1.0)).exp();
            let negative = (-EVSM_MAX_EXPONENT_RGBA16F * depth).exp();
            assert!(positive <= 1.0 && negative <= 1.0);
            assert!(positive.is_finite() && negative.is_finite());
        }
        let midpoint_second_moment = (-EVSM_MAX_EXPONENT_RGBA16F).exp();
        assert!(midpoint_second_moment >= 0.00006103515625);
        let endpoint = (2.0 * EVSM_MAX_EXPONENT_RGBA16F).exp();
        let endpoint_f16 = half::f16::from_f32(endpoint);
        assert!(endpoint <= f32::from(half::f16::MAX));
        assert!(endpoint_f16.is_finite());
        let negative_endpoint = (-2.0 * EVSM_MAX_EXPONENT_RGBA16F).exp();
        assert_ne!(half::f16::from_f32(negative_endpoint).to_bits(), 0);
    }

    #[test]
    fn shared_evsm_visibility_helpers_execute_lit_front_of_mean() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
visibility_output[0] = chebyshev_upper_bound_visibility(0.5, 0.01, 0.25);
visibility_output[1] = chebyshev_upper_bound_visibility(0.5, 0.01, 0.6);
let moments = vec4<f32>(0.5, 0.26, -0.5, 0.26);
visibility_output[2] =
    evsm_visibility_from_moments(moments, 0.25, -0.4, vec2<f32>(0.0001));
visibility_output[3] =
    evsm_visibility_from_moments(moments, 0.6, -0.75, vec2<f32>(0.0001));
visibility_output[4] =
    evsm_visibility_from_moments(moments, 0.25, -0.75, vec2<f32>(0.0001));";
        let Some([front, behind, positive_front, negative_front, both_front]) =
            execute_shader_probe(source, "EVSM visibility", probe)
        else {
            return;
        };
        assert_eq!(front, 1.0, "front-of-mean receiver must be lit");
        assert!(
            (behind - 0.5).abs() < 1.0e-5,
            "behind-mean Chebyshev result was {behind}"
        );
        assert!(
            positive_front.abs() < 1.0e-5,
            "dual-lobe bleed reduction did not preserve the negative lobe: {positive_front}"
        );
        assert!(
            negative_front.abs() < 1.0e-5,
            "dual-lobe bleed reduction did not preserve the positive lobe: {negative_front}"
        );
        assert_eq!(both_front, 1.0, "both front-of-mean lobes must be lit");
    }

    #[test]
    fn shared_evsm_minimum_variance_scales_each_warp_derivative() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let minimum = evsm_minimum_variance(
    vec2<f32>(2.0, -0.5),
    vec2<f32>(4.0, 4.0)
);
visibility_output[0] = minimum.x;
visibility_output[1] = minimum.y;";
        let Some([positive, negative, ..]) =
            execute_shader_probe(source, "EVSM minimum variance", probe)
        else {
            return;
        };
        assert!((positive - 0.000009).abs() < 1.0e-8);
        assert!((negative - 0.0000005625).abs() < 1.0e-9);
    }

    #[test]
    fn shared_evsm_moment_leak_control_preserves_a_soft_penumbra() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let exponent = 5.5;
let near_depth = exp(exponent * (0.45 - 1.0));
let far_depth = exp(exponent * (0.55 - 1.0));
let mean = 0.5 * (near_depth + far_depth);
let mean_squared = 0.5 * (near_depth * near_depth + far_depth * far_depth);
let moments = vec2<f32>(mean, mean_squared);
visibility_output[0] = evsm_moment_leak_control(
    moments, exp(exponent * (0.508 - 1.0)), exponent, 0.000001
);
visibility_output[1] = evsm_moment_leak_control(
    moments, exp(exponent * (0.511 - 1.0)), exponent, 0.000001
);
visibility_output[2] = evsm_moment_leak_control(
    moments, exp(exponent * (0.514 - 1.0)), exponent, 0.000001
);
visibility_output[3] = evsm_moment_leak_control(
    moments, exp(exponent * (0.517 - 1.0)), exponent, 0.000001
);
visibility_output[4] = evsm_moment_leak_control(
    moments, exp(exponent * (0.520 - 1.0)), exponent, 0.000001
);";
        let Some(curve) = execute_shader_probe(source, "EVSM moment leak control", probe) else {
            return;
        };
        assert!(
            curve.windows(2).all(|pair| pair[0] > pair[1]),
            "mixed-moment visibility is not a decreasing curve: {curve:?}"
        );
        assert!(
            curve[0] > 0.95,
            "mixed moments enter the transition too early: {curve:?}"
        );
        assert!(
            curve[4] < 0.05,
            "mixed moments did not converge to shadow: {curve:?}"
        );
        let soft_indices = curve
            .iter()
            .enumerate()
            .filter_map(|(index, &visibility)| {
                (visibility > 0.05 && visibility < 0.95).then_some(index)
            })
            .collect::<Vec<_>>();
        assert!(
            soft_indices.len() >= 3,
            "mixed moments collapsed to fewer than three soft samples: {curve:?}"
        );
        let transition_width =
            (soft_indices[soft_indices.len() - 1] - soft_indices[0]) as f32 * 0.003;
        assert!(
            transition_width >= 0.006,
            "mixed-moment transition collapsed below 0.006 depth: width={transition_width}, curve={curve:?}"
        );
    }

    #[test]
    fn shared_evsm_uniform_distribution_does_not_widen() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let exponent = 5.5;
let uniform_depth = exp(exponent * (0.5 - 1.0));
visibility_output[0] = evsm_moment_leak_control(
    vec2<f32>(uniform_depth, uniform_depth * uniform_depth),
    exp(exponent * (0.499 - 1.0)),
    exponent,
    0.000001
);
visibility_output[1] = evsm_moment_leak_control(
    vec2<f32>(uniform_depth, uniform_depth * uniform_depth),
    exp(exponent * (0.55 - 1.0)),
    exponent,
    0.000001
);";
        let Some([front, shadow, ..]) =
            execute_shader_probe(source, "EVSM uniform moment transition", probe)
        else {
            return;
        };
        assert_eq!(front, 1.0, "variance floor widened a uniform distribution");
        assert!(
            shadow < 0.05,
            "uniform moments leaked behind their occluder: {shadow}"
        );
    }

    #[test]
    fn shared_evsm_uniform_rgba16float_moments_remain_lit_at_the_receiver() {
        let depths = [0.1, 0.45, 0.5, 0.7, 0.9];
        let Some(visibility) = execute_evsm_half_uniform_probe(&depths) else {
            return;
        };
        for (depth, value) in depths.into_iter().zip(visibility) {
            assert_eq!(
                value, 1.0,
                "uniform moments falsely shadowed their receiver at depth {depth}: {value}"
            );
        }
    }

    #[test]
    fn shared_evsm_chebyshev_is_conservative_for_uniform_half_moments() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let receiver = 0.0111089965;
let moments = vec4<f32>(
    0.0111083984, 0.000123381615, -0.0111083984, 0.000123381615
);
let minimum = vec2<f32>(0.0000000014057148);
visibility_output[0] =
    evsm_visibility_from_moments(moments, receiver, -receiver, minimum);";
        let Some([visibility, ..]) =
            execute_shader_probe(source, "EVSM conservative half Chebyshev", probe)
        else {
            return;
        };
        assert_eq!(
            visibility, 1.0,
            "half rounding made a uniform receiver self-shadow: {visibility}"
        );
    }

    #[test]
    fn shared_evsm_uniform_half_uncertainty_has_a_bounded_transition() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let moments = vec2<f32>(0.0111083984, 0.000123381615);
visibility_output[0] =
    evsm_moment_leak_control(moments, 0.0111591, 5.5, 0.00000000146);
visibility_output[1] =
    evsm_moment_leak_control(moments, 0.017422374, 5.5, 0.00000000346);";
        let Some([near, far, ..]) =
            execute_shader_probe(source, "EVSM half-uncertainty transition", probe)
        else {
            return;
        };
        assert!(
            near > 0.05 && near < 0.95,
            "half uncertainty produced a hard near-receiver transition: {near}"
        );
        assert!(
            far < 0.05,
            "half uncertainty leaked far behind a uniform occluder: {far}"
        );
    }

    #[test]
    fn shared_evsm_uniform_rgba16float_dense_sweep_has_no_false_shadows() {
        let depths = (0..=10_000)
            .map(|i| i as f32 / 10_000.0)
            .collect::<Vec<_>>();
        let Some(visibility) = execute_evsm_half_uniform_probe(&depths) else {
            return;
        };
        let false_shadows = visibility.iter().filter(|&&value| value != 1.0).count();
        let minimum_visibility = visibility.iter().copied().fold(1.0, f32::min);
        assert_eq!(
            false_shadows, 0,
            "{false_shadows} of 10001 uniform depths were falsely shadowed"
        );
        assert_eq!(minimum_visibility, 1.0);
    }

    #[test]
    fn shared_evsm_helpers_fail_open_for_non_finite_inputs() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let nan_value = bitcast<f32>(0x7fc00000u);
let infinity = bitcast<f32>(0x7f800000u);
visibility_output[0] = evsm_visibility_from_moments(
    vec4<f32>(nan_value), 0.5, -0.5, vec2<f32>(0.0001)
);
visibility_output[1] = evsm_visibility_from_moments(
    vec4<f32>(0.5, 0.26, -0.5, 0.26), nan_value, -0.5, vec2<f32>(0.0001)
);
visibility_output[2] = evsm_visibility_from_moments(
    vec4<f32>(0.5, 0.26, -0.5, 0.26), 0.6, -0.6, vec2<f32>(nan_value)
);
visibility_output[3] = min(
    evsm_moment_leak_control(vec2<f32>(infinity), 0.5, 5.5, 0.0001),
    evsm_moment_leak_control(vec2<f32>(0.5, 0.26), 0.5, 5.5, nan_value)
);
visibility_output[4] = min(
    evsm_moment_leak_control(vec2<f32>(0.5, 0.26), infinity, 5.5, 0.0001),
    evsm_moment_leak_control(vec2<f32>(0.5, 0.26), 0.5, infinity, 0.0001)
);";
        let Some(visibility) = execute_shader_probe(source, "EVSM non-finite inputs", probe) else {
            return;
        };
        assert_eq!(visibility, [1.0; 5], "invalid EVSM inputs must fail open");
    }

    #[test]
    fn spatial_moment_blur_covers_every_moment_technique() {
        use crate::lighting::types::ShadowTechnique;

        assert!(requires_moment_blur(ShadowTechnique::VSM));
        assert!(requires_moment_blur(ShadowTechnique::MSM));
        assert!(requires_moment_blur(ShadowTechnique::EVSM));
        assert!(!requires_moment_blur(ShadowTechnique::PCF));
    }

    #[test]
    fn shared_pcss_penumbra_grows_with_light_size_and_receiver_separation() {
        let probe = "
visibility_output[0] = pcss_penumbra_size(0.55, 0.5, 1.0);
visibility_output[1] = pcss_penumbra_size(0.55, 0.5, 12.0);
visibility_output[2] = pcss_penumbra_size(0.8, 0.5, 1.0);
visibility_output[3] = pcss_penumbra_size(0.8, 0.5, 12.0);
visibility_output[4] = 0.0;";
        let Some([near_small, near_large, far_small, far_large, _]) =
            execute_shader_probe(CSM_SHADER_SOURCE, "PCSS penumbra", probe)
        else {
            return;
        };
        let near_growth = near_large - near_small;
        let far_growth = far_large - far_small;

        assert!(near_growth > 1.0, "light-size response was {near_growth}");
        assert!(
            far_growth >= near_growth + 4.0,
            "receiver separation did not widen PCSS: near={near_growth}, far={far_growth}"
        );
    }

    #[test]
    fn shared_msm_visibility_is_bounded_for_degenerate_and_non_finite_inputs() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let nan_value = bitcast<f32>(0x7fc00000u);
visibility_output[0] =
    msm_visibility_from_moments(vec4<f32>(0.5, 0.33333334, 0.25, 0.2), 0.6, 0.0005);
visibility_output[1] =
    msm_visibility_from_moments(vec4<f32>(0.5, 0.25, 0.125, 0.0625), 0.7, 0.0);
visibility_output[2] =
    msm_visibility_from_moments(vec4<f32>(nan_value), 0.7, 0.0005);
visibility_output[3] =
    msm_visibility_from_moments(vec4<f32>(0.5, 0.25, 0.125, 0.0625), nan_value, 0.0005);
visibility_output[4] =
    msm_visibility_from_moments(vec4<f32>(0.9, 0.1, 0.9, 0.1), 0.95, 0.0005);";
        let Some(values) = execute_shader_probe(source, "MSM boundary behavior", probe) else {
            return;
        };
        for value in values {
            assert!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "MSM returned invalid visibility {value}"
            );
        }
        assert_eq!(values[2], 1.0, "invalid stored moments must fail open");
        assert_eq!(values[3], 1.0, "invalid receiver depth must fail open");
    }
}

#[cfg(test)]
mod msm_tests;
