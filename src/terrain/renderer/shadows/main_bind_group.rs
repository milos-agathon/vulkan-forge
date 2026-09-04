use super::*;
use crate::core::resource_tracker::tracked_create_buffer_init;

impl TerrainScene {
    pub(super) fn create_shadow_bind_group(&self) -> Result<wgpu::BindGroup> {
        use crate::core::shadow_mapping::{CsmCascadeData, CsmUniforms};

        let csm = &self.csm_renderer.uniforms;
        let terrain_csm_uniforms = CsmUniforms {
            light_direction: csm.light_direction,
            light_view: [
                [
                    csm.light_view[0],
                    csm.light_view[1],
                    csm.light_view[2],
                    csm.light_view[3],
                ],
                [
                    csm.light_view[4],
                    csm.light_view[5],
                    csm.light_view[6],
                    csm.light_view[7],
                ],
                [
                    csm.light_view[8],
                    csm.light_view[9],
                    csm.light_view[10],
                    csm.light_view[11],
                ],
                [
                    csm.light_view[12],
                    csm.light_view[13],
                    csm.light_view[14],
                    csm.light_view[15],
                ],
            ],
            cascades: {
                fn flat_to_2d(arr: &[f32; 16]) -> [[f32; 4]; 4] {
                    [
                        [arr[0], arr[1], arr[2], arr[3]],
                        [arr[4], arr[5], arr[6], arr[7]],
                        [arr[8], arr[9], arr[10], arr[11]],
                        [arr[12], arr[13], arr[14], arr[15]],
                    ]
                }
                [
                    CsmCascadeData {
                        light_projection: flat_to_2d(&csm.cascades[0].light_projection),
                        light_view_proj: csm.cascades[0].light_view_proj,
                        near_distance: csm.cascades[0].near_distance,
                        far_distance: csm.cascades[0].far_distance,
                        texel_size: csm.cascades[0].texel_size,
                        _padding: 0.0,
                    },
                    CsmCascadeData {
                        light_projection: flat_to_2d(&csm.cascades[1].light_projection),
                        light_view_proj: csm.cascades[1].light_view_proj,
                        near_distance: csm.cascades[1].near_distance,
                        far_distance: csm.cascades[1].far_distance,
                        texel_size: csm.cascades[1].texel_size,
                        _padding: 0.0,
                    },
                    CsmCascadeData {
                        light_projection: flat_to_2d(&csm.cascades[2].light_projection),
                        light_view_proj: csm.cascades[2].light_view_proj,
                        near_distance: csm.cascades[2].near_distance,
                        far_distance: csm.cascades[2].far_distance,
                        texel_size: csm.cascades[2].texel_size,
                        _padding: 0.0,
                    },
                    CsmCascadeData {
                        light_projection: flat_to_2d(&csm.cascades[3].light_projection),
                        light_view_proj: csm.cascades[3].light_view_proj,
                        near_distance: csm.cascades[3].near_distance,
                        far_distance: csm.cascades[3].far_distance,
                        texel_size: csm.cascades[3].texel_size,
                        _padding: 0.0,
                    },
                ]
            },
            cascade_count: csm.cascade_count,
            pcf_kernel_size: csm.pcf_kernel_size,
            depth_bias: csm.depth_bias,
            slope_bias: csm.slope_bias,
            shadow_map_size: csm.shadow_map_size,
            debug_mode: csm.debug_mode,
            // Match MomentGenerationPass::execute's clamp for the Rgba16Float atlas.
            evsm_positive_exp: crate::shadows::clamp_evsm_exponent(csm.evsm_positive_exp),
            evsm_negative_exp: crate::shadows::clamp_evsm_exponent(csm.evsm_negative_exp),
            peter_panning_offset: csm.peter_panning_offset,
            enable_unclipped_depth: csm.enable_unclipped_depth,
            depth_clip_factor: csm.depth_clip_factor,
            technique: self.shadow_technique,
            technique_flags: csm.technique_flags,
            _padding1: [0.0; 3],
            technique_params: csm.technique_params,
            technique_reserved: csm.technique_reserved,
            cascade_blend_range: csm.cascade_blend_range,
            _padding2: [0.0; 27],
        };

        let terrain_csm_buffer = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.shadow.csm_uniforms"),
                contents: bytemuck::bytes_of(&terrain_csm_uniforms),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;

        let shadow_texture_view = self.csm_renderer.shadow_texture_view();
        let moment_texture_view = self.csm_renderer.moment_texture_view();
        let moment_view_ref = moment_texture_view
            .as_ref()
            .unwrap_or(&self.noop_shadow.moment_maps_view);

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.shadow.main_bind_group"),
            layout: &self.shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: terrain_csm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.csm_renderer.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(moment_view_ref),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.noop_shadow.moment_sampler),
                },
            ],
        }))
    }
}
