use super::super::{CsmRenderer, CsmUniforms, MomentGenerationPass, ShadowBlurPass};
use super::budget;
use super::types::*;
use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::lighting::types::ShadowTechnique;
use glam::{Mat4, Vec3};
use std::num::NonZeroU64;
use wgpu::{
    AddressMode, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, Device, Extent3d, FilterMode, Queue, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderStages, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

/// Runtime controller that keeps the CSM atlas and technique uniforms in sync.
pub struct ShadowManager {
    config: ShadowManagerConfig,
    renderer: CsmRenderer,
    moment_sampler: Sampler,
    fallback_moment_texture: Option<TrackedTexture>,
    moment_pass: Option<MomentGenerationPass>,
    /// P0.2/M3: Blur pass for VSM/MSM soft shadows
    blur_pass: Option<ShadowBlurPass>,
    requires_moments: bool,
    memory_bytes: u64,
}

impl ShadowManager {
    /// Construct manager while respecting memory constraints and technique requirements.
    pub fn new(device: &Device, mut config: ShadowManagerConfig) -> RenderResult<Self> {
        let requires_moments = matches!(
            config.technique,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        );
        config.csm.enable_evsm = requires_moments;

        budget::enforce_memory_budget(&mut config)?;

        let renderer = CsmRenderer::new(device, config.csm.clone())?;

        let moment_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_moment_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        let fallback_moment_texture = if requires_moments {
            None
        } else {
            Some(Self::create_fallback_moment_texture(device)?)
        };

        let memory_bytes = budget::estimate_memory_bytes(
            config.csm.shadow_map_size,
            config.csm.cascade_count,
            config.technique,
        )?;

        // Create moment generation and blur passes if needed
        let moment_pass = if requires_moments {
            Some(MomentGenerationPass::new(device)?)
        } else {
            None
        };

        let blur_pass = if super::super::requires_moment_blur(config.technique) {
            Some(ShadowBlurPass::new(device)?)
        } else {
            None
        };

        let mut manager = Self {
            config,
            renderer,
            moment_sampler,
            fallback_moment_texture,
            moment_pass,
            blur_pass,
            requires_moments,
            memory_bytes,
        };

        manager.apply_uniform_overrides();
        Ok(manager)
    }

    /// Access underlying configuration.
    pub fn config(&self) -> &ShadowManagerConfig {
        &self.config
    }

    /// Mutable access to configuration for dynamic adjustments.
    pub fn config_mut(&mut self) -> &mut ShadowManagerConfig {
        &mut self.config
    }

    /// Underlying cascaded shadow renderer.
    pub fn renderer(&self) -> &CsmRenderer {
        &self.renderer
    }

    /// Mutable renderer access.
    pub fn renderer_mut(&mut self) -> &mut CsmRenderer {
        &mut self.renderer
    }

    /// Depth sampler used for comparisons.
    pub fn shadow_sampler(&self) -> &Sampler {
        &self.renderer.shadow_sampler
    }

    /// Sampler used for moment-based filtering (also bound for fallbacks).
    pub fn moment_sampler(&self) -> &Sampler {
        &self.moment_sampler
    }

    /// Get shadow map resolution (P3-12)
    pub fn shadow_map_size(&self) -> u32 {
        self.config.csm.shadow_map_size
    }

    /// Get cascade count (P3-12)
    pub fn cascade_count(&self) -> u32 {
        self.config.csm.cascade_count
    }

    /// Get debug info string for logging (P3-12)
    pub fn debug_info(&self) -> String {
        let technique_name = match self.config.technique {
            ShadowTechnique::Hard => "Hard",
            ShadowTechnique::PCF => "PCF",
            ShadowTechnique::PCSS => "PCSS",
            ShadowTechnique::VSM => "VSM",
            ShadowTechnique::EVSM => "EVSM",
            ShadowTechnique::MSM => "MSM",
            ShadowTechnique::CSM => "CSM (PCF)",
        };

        let memory_mib = self.memory_bytes as f64 / (1024.0 * 1024.0);

        format!(
            "Shadow Manager Configuration:\n\
             - Technique: {}\n\
             - Shadow Map Size: {}x{}\n\
             - Cascade Count: {}\n\
             - Total Memory: {:.2} MiB\n\
             - PCSS Blocker Radius (texels): {:.4}\n\
             - PCSS Filter Radius (texels): {:.4}\n\
             - Light Size: {:.4}\n\
             - Moment Bias: {:.6}\n\
             - Requires Moments: {}",
            technique_name,
            self.config.csm.shadow_map_size,
            self.config.csm.shadow_map_size,
            self.config.csm.cascade_count,
            memory_mib,
            self.config.pcss_blocker_radius,
            self.config.pcss_filter_radius,
            self.config.light_size,
            self.config.moment_bias,
            self.requires_moments,
        )
    }

    /// Get cascade debug info after cascades have been updated (P3-12)
    pub fn cascade_debug_info(&self) -> String {
        let uniforms = &self.renderer.uniforms;
        let mut info = String::from("Cascade Details:\n");

        for i in 0..uniforms.cascade_count as usize {
            if i < 4 {
                let cascade = &uniforms.cascades[i];
                info.push_str(&format!(
                    "  Cascade {}: near={:.2}, far={:.2}, texel_size={:.6}\n",
                    i, cascade.near_distance, cascade.far_distance, cascade.texel_size
                ));
            }
        }

        info
    }

    /// Create a bind group layout that matches the active shadow technique.
    pub fn create_bind_group_layout(&self, device: &Device) -> BindGroupLayout {
        let entries = [
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<CsmUniforms>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Comparison),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ];

        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_manager_bind_group_layout"),
            entries: &entries,
        })
    }

    /// Depth atlas view covering all cascades.
    pub fn shadow_view(&self) -> wgpu::TextureView {
        self.renderer.shadow_texture_view()
    }

    /// Moment atlas view or a fallback texture when moments are unused.
    pub fn moment_view(&self) -> wgpu::TextureView {
        if let Some(view) = self.renderer.moment_texture_view() {
            view
        } else if let Some(ref texture) = self.fallback_moment_texture {
            let layer_count = self.config.csm.cascade_count.max(1).min(4);
            texture.create_view(&TextureViewDescriptor {
                label: Some("shadow_moment_fallback_view"),
                format: Some(TextureFormat::Rgba16Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(layer_count),
                ..Default::default()
            })
        } else {
            self.renderer.shadow_texture_view()
        }
    }

    /// Active technique enumeration.
    pub fn technique(&self) -> ShadowTechnique {
        self.config.technique
    }

    /// Total GPU memory currently consumed by the atlas.
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Update cascades with latest camera/light state and refresh technique parameters.
    pub fn update_cascades(
        &mut self,
        camera_view: Mat4,
        camera_projection: Mat4,
        light_direction: Vec3,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.renderer.update_cascades(
            camera_view,
            camera_projection,
            light_direction,
            near_plane,
            far_plane,
        );
        self.apply_uniform_overrides();
    }

    /// Upload uniforms to GPU.
    pub fn upload_uniforms(&self, queue: &Queue) {
        self.renderer.upload_uniforms(queue);
    }

    /// Populate moment maps from depth maps (VSM/EVSM/MSM only).
    /// Call this after rendering shadow depth maps for all cascades.
    pub fn populate_moments(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderResult<()> {
        // Only execute if technique requires moments
        if !self.requires_moments {
            return Ok(());
        }

        let moment_pass = match &mut self.moment_pass {
            Some(pass) => pass,
            None => return Ok(()),
        };

        let moment_texture = match &self.renderer.evsm_maps {
            Some(tex) => tex,
            None => return Ok(()),
        };

        // Prepare bind group
        moment_pass.prepare_textures_checked(
            device,
            self.renderer.shadow_maps.as_ref(),
            moment_texture,
        )?;

        // Execute moment generation compute pass
        moment_pass.execute_checked(
            queue,
            encoder,
            self.config.technique,
            self.config.csm.cascade_count,
            self.config.csm.shadow_map_size,
            self.config.csm.evsm_positive_exp,
            self.config.csm.evsm_negative_exp,
        )?;

        // Apply Gaussian blur to all moment techniques.
        if let Some(blur_pass) = &mut self.blur_pass {
            blur_pass.execute(
                device,
                queue,
                encoder,
                moment_texture,
                self.config.csm.cascade_count,
                self.config.csm.shadow_map_size,
                self.config.blur_kernel_radius,
                self.config.technique,
                self.config.csm.evsm_positive_exp,
            )?;
        }

        Ok(())
    }

    /// Returns true if the active technique reads the moment atlas.
    pub fn uses_moments(&self) -> bool {
        self.requires_moments
    }

    /// Re-apply technique parameters after external configuration tweaks.
    pub fn refresh_uniforms(&mut self) {
        self.apply_uniform_overrides();
    }

    fn apply_uniform_overrides(&mut self) {
        self.renderer.set_debug_mode(self.config.csm.debug_mode);
        self.renderer.uniforms.technique = self.config.technique.as_u32();
        self.renderer.uniforms.technique_flags =
            Self::compute_flags(self.config.technique, self.requires_moments);
        self.renderer.uniforms.technique_params = [
            self.config.pcss_blocker_radius,
            self.config.pcss_filter_radius,
            self.config.moment_bias,
            self.config.light_size,
        ];
        self.renderer.uniforms.technique_reserved = [0.0; 4];

        if matches!(self.config.technique, ShadowTechnique::PCSS) {
            self.clamp_pcss_radii();
        }
    }

    fn clamp_pcss_radii(&mut self) {
        self.renderer.uniforms.technique_params[0] = self
            .config
            .pcss_blocker_radius
            .min(super::types::MAX_PCSS_BLOCKER_RADIUS_TEXELS);
        self.renderer.uniforms.technique_params[1] = self
            .config
            .pcss_filter_radius
            .min(super::types::MAX_PCSS_FILTER_RADIUS_TEXELS);
    }

    fn compute_flags(technique: ShadowTechnique, requires_moments: bool) -> u32 {
        let mut flags = 0u32;
        if requires_moments {
            flags |= 0b01;
        }
        if matches!(technique, ShadowTechnique::PCSS) {
            flags |= 0b10;
        }
        flags
    }

    fn create_fallback_moment_texture(device: &Device) -> RenderResult<TrackedTexture> {
        tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("shadow_moment_fallback"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 4,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )
    }
}
