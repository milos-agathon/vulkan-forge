use super::*;

#[test]
fn shadows_require_power_of_two_map() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.map_size = 300;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("power of two"));
}

#[test]
fn shadows_map_size_must_be_nonzero() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.map_size = 0;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("greater than zero"));
}

#[test]
fn hard_shadows_preserve_small_power_of_two_maps() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Hard;
    cfg.shadows.map_size = 128;
    cfg.validate().expect("small hard-shadow atlas");
}

#[test]
fn filtered_shadows_preserve_256_maps() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcf;
    cfg.shadows.map_size = 256;
    cfg.validate().expect("256px filtered atlas");
}

#[test]
fn shadows_cascades_must_be_in_range() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.cascades = 0;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("[1, 4]"));

    cfg.shadows.cascades = 5;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("[1, 4]"));
}

#[test]
fn shadows_csm_requires_multiple_cascades() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Csm;
    cfg.shadows.cascades = 1;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains(">= 2"));
}

#[test]
fn shadows_pcss_requires_positive_blocker_radius() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcss;
    cfg.shadows.pcss_blocker_radius = -0.1;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("pcss_blocker_radius"));
}

#[test]
fn shadows_pcss_requires_positive_filter_radius() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcss;
    cfg.shadows.pcss_filter_radius = -0.1;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("pcss_filter_radius"));
}

#[test]
fn shadows_pcss_requires_positive_light_size() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcss;
    cfg.shadows.light_size = 0.0;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("light_size"));
}

#[test]
fn shadows_moment_techniques_require_positive_bias() {
    for technique in [
        ShadowTechnique::Vsm,
        ShadowTechnique::Evsm,
        ShadowTechnique::Msm,
    ] {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = technique;
        cfg.shadows.moment_bias = 0.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("moment_bias"));
    }
}

#[test]
fn shadow_memory_budget_is_enforced() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Evsm;
    cfg.shadows.map_size = 8192;
    cfg.shadows.cascades = 4;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("256 MiB"));
}

#[test]
fn shadows_filtered_techniques_recommend_min_resolution() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcf;
    cfg.shadows.map_size = 128;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("at least 256"));
}
