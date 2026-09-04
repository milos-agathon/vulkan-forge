use super::*;

impl PbrPipelineWithShadows {
    pub fn has_shadows(&self) -> bool {
        self.shadow_manager.is_some()
    }

    pub fn get_cascade_info(&self, cascade_idx: usize) -> Option<(f32, f32, f32)> {
        self.shadow_manager
            .as_ref()
            .and_then(|mgr| mgr.renderer().get_cascade_info(cascade_idx))
    }

    pub fn validate_peter_panning_prevention(&self) -> bool {
        self.shadow_manager
            .as_ref()
            .map(|mgr| mgr.renderer().validate_peter_panning_prevention())
            .unwrap_or(true)
    }

    pub fn shadow_layout(&self) -> Option<&BindGroupLayout> {
        self.shadow_bind_group_layout.as_ref()
    }

    pub(super) fn rebuild_shadow_resources(&mut self, device: &Device) {
        let manager = match ShadowManager::new(device, self.shadow_config.clone()) {
            Ok(manager) => manager,
            Err(e) => {
                log::error!(target: "pbr.shadow", "failed to rebuild shadow resources: {e}");
                crate::core::degradation::record_degradation(
                    "allocation_fallback",
                    "pbr.shadow_resources",
                    "shadow resources not rebuilt; PBR pipeline renders without shadows",
                );
                return;
            }
        };
        self.shadow_config = manager.config().clone();
        let layout = manager.create_bind_group_layout(device);
        self.shadow_bind_group = None;
        self.shadow_bind_group_layout = Some(layout);
        self.shadow_manager = Some(manager);
        self.render_pipeline = None;
        self.pipeline_format = None;
    }

    pub(super) fn drop_shadow_resources(&mut self) {
        self.shadow_manager = None;
        self.shadow_bind_group = None;
        self.shadow_bind_group_layout = None;
        self.render_pipeline = None;
        self.pipeline_format = None;
    }
}

/// Create shadow manager with predefined quality presets
pub fn create_csm_with_preset(
    device: &Device,
    preset: CsmQualityPreset,
) -> crate::core::error::RenderResult<ShadowManager> {
    let mut config = ShadowManagerConfig::default();

    match preset {
        CsmQualityPreset::Low => {
            config.csm.cascade_count = 3;
            config.csm.shadow_map_size = 1024;
            config.csm.pcf_kernel_size = 1;
            config.csm.depth_bias = 0.01;
            config.csm.slope_bias = 0.02;
            config.csm.peter_panning_offset = 0.002;
            config.technique = ShadowTechnique::Hard;
        }
        CsmQualityPreset::Medium => {
            config.csm.cascade_count = 3;
            config.csm.shadow_map_size = 2048;
            config.csm.pcf_kernel_size = 3;
            config.csm.depth_bias = 0.005;
            config.csm.slope_bias = 0.01;
            config.csm.peter_panning_offset = 0.001;
            config.technique = ShadowTechnique::PCF;
        }
        CsmQualityPreset::High => {
            config.csm.cascade_count = 4;
            config.csm.shadow_map_size = 4096;
            config.csm.pcf_kernel_size = 5;
            config.csm.depth_bias = 0.003;
            config.csm.slope_bias = 0.005;
            config.csm.peter_panning_offset = 0.0005;
            config.technique = ShadowTechnique::PCF;
        }
        CsmQualityPreset::Ultra => {
            config.csm.cascade_count = 4;
            config.csm.shadow_map_size = 4096;
            config.csm.pcf_kernel_size = 7;
            config.csm.depth_bias = 0.002;
            config.csm.slope_bias = 0.003;
            config.csm.peter_panning_offset = 0.0003;
            config.technique = ShadowTechnique::EVSM;
            config.pcss_blocker_radius = 0.02;
            config.pcss_filter_radius = 0.05;
            config.moment_bias = 0.0002;
        }
    };

    ShadowManager::new(device, config)
}

/// CSM quality presets for different performance/quality tradeoffs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsmQualityPreset {
    /// Low quality: 3 cascades, 1024px, no PCF
    Low,
    /// Medium quality: 3 cascades, 2048px, 3x3 PCF
    Medium,
    /// High quality: 4 cascades, 4096px, 5x5 PCF
    High,
    /// Ultra quality: 4 cascades, 4096px, Poisson PCF + EVSM
    Ultra,
}

/// Return the complete shared CSM WGSL module.
pub fn csm_shader_source() -> &'static str {
    crate::shadows::CsmRenderer::shader_source()
}

#[cfg(test)]
mod tests {
    #[test]
    fn public_csm_shader_source_returns_valid_wgsl() {
        let source = super::csm_shader_source();
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    }
}
