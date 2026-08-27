// src/viewer/terrain/render.rs
// Terrain rendering for the interactive viewer

use super::scene::ViewerTerrainScene;
use crate::lighting::shadow::ShadowTechnique;
use crate::shadows::CsmUniforms;

/// Shader for accumulating frames (additive blend)
const ACCUMULATE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_tex, samp, in.uv);
}
"#;

/// Terrain uniforms for the simple terrain shader
#[repr(C, align(16))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TerrainUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub terrain_params: [f32; 4],
    pub lighting: [f32; 4],
    pub background: [f32; 4],
    pub water_color: [f32; 4],
    pub render_origin_xz: [f32; 2],
    pub render_span_xz: [f32; 2],
}

/// Shadow pass uniforms for depth-only terrain rendering (per cascade)
/// Must match ShadowPassUniforms in terrain_shadow_depth.wgsl exactly (128 bytes)
#[repr(C, align(16))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowPassUniforms {
    pub light_view_proj: [[f32; 4]; 4], // 64 bytes
    pub terrain_params: [f32; 4],       // 16 bytes: spacing, height_exag, height_min, height_max
    pub grid_params: [f32; 4],          // 16 bytes: grid_resolution, _pad, _pad, _pad
    pub height_curve: [f32; 4],         // 16 bytes: mode, strength, power, _pad
    pub render_origin_xz: [f32; 2],     // offset 112
    pub render_span_xz: [f32; 2],       // offset 120
}

/// Extended uniforms for PBR terrain shader
#[repr(C, align(16))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TerrainPbrUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub terrain_params: [f32; 4],
    pub lighting: [f32; 4],
    pub background: [f32; 4],
    pub water_color: [f32; 4],
    pub pbr_params: [f32; 4], // exposure, normal_strength, ibl_intensity, overlay_preserve_colors
    pub ibl_params: [f32; 4], // use_hdri (>0.5), specular_max_mip, sin_theta, cos_theta
    pub camera_pos: [f32; 4], // camera world position
    pub lens_params: [f32; 4], // vignette_strength, vignette_radius, vignette_softness, _
    pub screen_dims: [f32; 4], // width, height, media near, media far
    pub overlay_params: [f32; 4], // enabled (>0.5), opacity, blend_mode (0=normal, 1=multiply, 2=overlay), solid (>0.5)
    pub render_origin_xz: [f32; 2],
    pub render_span_xz: [f32; 2],
}

const _: [(); 160] = [(); std::mem::size_of::<TerrainUniforms>()];
const _: [(); 16] = [(); std::mem::align_of::<TerrainUniforms>()];
const _: [(); 144] = [(); std::mem::offset_of!(TerrainUniforms, render_origin_xz)];
const _: [(); 152] = [(); std::mem::offset_of!(TerrainUniforms, render_span_xz)];
const _: [(); 128] = [(); std::mem::size_of::<ShadowPassUniforms>()];
const _: [(); 16] = [(); std::mem::align_of::<ShadowPassUniforms>()];
const _: [(); 112] = [(); std::mem::offset_of!(ShadowPassUniforms, render_origin_xz)];
const _: [(); 120] = [(); std::mem::offset_of!(ShadowPassUniforms, render_span_xz)];
const _: [(); 256] = [(); std::mem::size_of::<TerrainPbrUniforms>()];
const _: [(); 16] = [(); std::mem::align_of::<TerrainPbrUniforms>()];
const _: [(); 240] = [(); std::mem::offset_of!(TerrainPbrUniforms, render_origin_xz)];
const _: [(); 248] = [(); std::mem::offset_of!(TerrainPbrUniforms, render_span_xz)];

#[cfg(test)]
mod terrain_uniform_abi_tests {
    use super::{ShadowPassUniforms, TerrainPbrUniforms, TerrainUniforms};

    #[test]
    fn append_only_uniform_sizes_and_offsets_match_the_m06_contract() {
        assert_eq!(std::mem::size_of::<TerrainUniforms>(), 160);
        assert_eq!(std::mem::align_of::<TerrainUniforms>(), 16);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, render_origin_xz), 144);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, render_span_xz), 152);

        assert_eq!(std::mem::size_of::<ShadowPassUniforms>(), 128);
        assert_eq!(std::mem::align_of::<ShadowPassUniforms>(), 16);
        assert_eq!(
            std::mem::offset_of!(ShadowPassUniforms, render_origin_xz),
            112
        );
        assert_eq!(
            std::mem::offset_of!(ShadowPassUniforms, render_span_xz),
            120
        );

        assert_eq!(std::mem::size_of::<TerrainPbrUniforms>(), 256);
        assert_eq!(std::mem::align_of::<TerrainPbrUniforms>(), 16);
        assert_eq!(
            std::mem::offset_of!(TerrainPbrUniforms, render_origin_xz),
            240
        );
        assert_eq!(
            std::mem::offset_of!(TerrainPbrUniforms, render_span_xz),
            248
        );
    }
}

impl ViewerTerrainScene {
    /// Canonical media always consumes and produces linear HDR. Legacy viewer
    /// paths keep their established output format.
    pub(super) fn scene_color_format(&self) -> wgpu::TextureFormat {
        if self.canonical_media.is_some() {
            wgpu::TextureFormat::Rgba16Float
        } else {
            self.surface_format
        }
    }

    pub(super) fn scene_color_format_for(
        &self,
        output_format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormat {
        if self.canonical_media.is_some() {
            wgpu::TextureFormat::Rgba16Float
        } else {
            output_format
        }
    }

    pub(super) fn ensure_depth(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if self.depth_size != (width, height) {
            let tex = crate::core::resource_tracker::tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.depth"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )?;
            // Create depth view with explicit DepthOnly aspect for sampling
            self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain_viewer.depth_view"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            }));
            self.depth_texture = Some(tex);
            self.depth_size = (width, height);
        }
        Ok(())
    }
}

mod helpers;
mod motion_blur;
mod offscreen;
mod screen;
mod shadow;
