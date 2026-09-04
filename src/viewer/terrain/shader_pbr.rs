// src/viewer/terrain/shader_pbr.rs
// Enhanced PBR-like shader for terrain rendering with better lighting

pub const TERRAIN_PBR_SHADER: &str = concat!(
    include_str!("../../shaders/includes/shadow_moments.wgsl"),
    "\n",
    include_str!("shader_pbr/terrain_pbr.wgsl")
);

pub(crate) fn default_viewer_pcss_technique_params() -> [f32; 4] {
    [
        crate::shadows::DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS,
        crate::shadows::DEFAULT_PCSS_FILTER_RADIUS_TEXELS,
        0.0005,
        crate::shadows::DEFAULT_PCSS_LIGHT_SIZE,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_shader_has_valid_blocker_dependent_pcss_dispatch() {
        let module = naga::front::wgsl::parse_str(TERRAIN_PBR_SHADER)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(TERRAIN_PBR_SHADER)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(TERRAIN_PBR_SHADER)));

        assert!(TERRAIN_PBR_SHADER.contains("if (technique == 2u)"));
        assert!(TERRAIN_PBR_SHADER.contains("pcss_blocker_search("));
        assert!(TERRAIN_PBR_SHADER.contains("avg_blocker_depth"));
        assert!(TERRAIN_PBR_SHADER.contains("csm_uniforms.technique_params.w"));
    }

    #[test]
    fn viewer_uploads_nonzero_pcss_defaults() {
        assert_eq!(
            default_viewer_pcss_technique_params(),
            [6.0, 4.0, 0.0005, 1.0]
        );
    }
}
