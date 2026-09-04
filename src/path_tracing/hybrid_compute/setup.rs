use super::*;

impl HybridPathTracer {
    pub fn new() -> Result<Self, RenderError> {
        let device = &try_ctx()?.device;
        let shader_src = crate::shader_sources::hybrid_kernel();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "hybrid-pt-kernel",
            &shader_src,
        );

        let layouts = HybridBindGroupLayouts {
            uniforms: Self::create_uniforms_layout(device),
            scene: Self::create_scene_layout(device),
            accum: Self::create_accum_layout(device),
            output: Self::create_output_layout(device),
            terrain_gbuffer: Self::create_terrain_gbuffer_layout(device),
            restir_temporal: Self::create_restir_temporal_layout(device),
            restir_spatial_scene: Self::create_restir_spatial_scene_layout(device),
            restir_spatial_reuse: Self::create_restir_spatial_reuse_layout(device),
            empty: Self::create_empty_layout(device),
        };

        // Four bind groups — the whole pipeline stays runnable on adapters
        // capped at max_bind_groups = 4 (lighting lives in group 0).
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-pipeline-layout"),
            bind_group_layouts: &[
                &layouts.uniforms,
                &layouts.scene,
                &layouts.accum,
                &layouts.output,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("hybrid-pt-compute"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        );
        let pipeline_terrain = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("hybrid-pt-terrain-compute"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main_terrain",
            },
        );
        let aether_reference_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hybrid-pt-aether-spectral-reference-layout"),
                bind_group_layouts: &[&layouts.uniforms, &layouts.scene, &layouts.accum],
                push_constant_ranges: &[],
            });
        let pipeline_aether_reference =
            crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("hybrid-pt-aether-spectral-reference-compute"),
                    layout: Some(&aether_reference_layout),
                    module: &shader,
                    entry_point: "main_aether_spectral_reference",
                },
            );

        // ReSTIR G-buffer entry: its group-2 variant carries the G-buffer
        // storage bindings so the main kernels stay within 8 storage buffers
        // per compute stage.
        let gbuffer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-terrain-gbuffer-layout"),
            bind_group_layouts: &[&layouts.uniforms, &layouts.scene, &layouts.terrain_gbuffer],
            push_constant_ranges: &[],
        });
        let pipeline_terrain_gbuffer = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("hybrid-pt-terrain-gbuffer-compute"),
                layout: Some(&gbuffer_layout),
                module: &shader,
                entry_point: "main_terrain_gbuffer",
            },
        );

        // Canonical ReSTIR reuse passes, compiled from the same WGSL the
        // wavefront scheduler uses so the reservoir layout stays one contract.
        let temporal_shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "hybrid-pt-restir-temporal",
            include_str!("../../shaders/pt_restir_temporal.wgsl"),
        );
        let temporal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-restir-temporal-layout"),
            bind_group_layouts: &[&layouts.uniforms, &layouts.empty, &layouts.restir_temporal],
            push_constant_ranges: &[],
        });
        let pipeline_restir_temporal = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("hybrid-pt-restir-temporal-compute"),
                layout: Some(&temporal_layout),
                module: &temporal_shader,
                entry_point: "main",
            },
        );

        let spatial_shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "hybrid-pt-restir-spatial",
            include_str!("../../shaders/pt_restir_spatial.wgsl"),
        );
        let spatial_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-restir-spatial-layout"),
            bind_group_layouts: &[
                &layouts.uniforms,
                &layouts.restir_spatial_scene,
                &layouts.restir_spatial_reuse,
            ],
            push_constant_ranges: &[],
        });
        let pipeline_restir_spatial = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("hybrid-pt-restir-spatial-compute"),
                layout: Some(&spatial_layout),
                module: &spatial_shader,
                entry_point: "main",
            },
        );

        Ok(Self {
            layouts,
            pipeline,
            pipeline_terrain,
            pipeline_aether_reference,
            pipeline_terrain_gbuffer,
            pipeline_restir_temporal,
            pipeline_restir_spatial,
        })
    }
}
