//! Renderer-owned WGSL assembly shared with the static verifier.

fn strip_includes(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn hybrid_kernel() -> String {
    [
        include_str!("shaders/sdf_primitives.wgsl").to_string(),
        strip_includes(include_str!("shaders/sdf_operations.wgsl")),
        strip_includes(include_str!("shaders/hybrid_traversal.wgsl")),
        strip_includes(include_str!("shaders/hybrid_terrain_traversal.wgsl")),
        include_str!("shaders/atmosphere/prometheus_spectral_reference.wgsl").to_string(),
        strip_includes(include_str!("shaders/hybrid_kernel.wgsl")),
    ]
    .join("\n")
}

/// AETHER sky module: the established camera/sky ABI plus the spectral LUT
/// evaluator in its dedicated bind group. The legacy sky module remains a
/// separate source and cannot accidentally claim AETHER shader provenance.
pub(crate) fn aether_sky() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/sky.wgsl").to_string(),
        include_str!("shaders/atmosphere/evaluation_core.wgsl").to_string(),
        include_str!("shaders/atmosphere/scattering.wgsl").to_string(),
    ]
    .join("\n")
}

/// Display-only AETHER resolve. Atmosphere evaluation itself stays linear HDR
/// and reaches the existing tonemap operator unchanged through this assembly.
pub(crate) fn aether_blit() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/includes/tonemap_common.wgsl").to_string(),
        include_str!("shaders/terrain_aether_blit.wgsl").to_string(),
    ]
    .join("\n")
}

/// Standalone PROMETHEUS aerial post. Keeping this source out of
/// `hybrid_kernel()` is the contract that the established traversal and
/// accumulation bind-group layouts remain byte-for-byte untouched.
pub(crate) fn prometheus_aerial() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/includes/tonemap_common.wgsl").to_string(),
        include_str!("shaders/atmosphere/evaluation_core.wgsl").to_string(),
        include_str!("shaders/atmosphere/prometheus_aerial.wgsl").to_string(),
    ]
    .join("\n")
}

pub(crate) fn terrain() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/atmosphere/evaluation_core.wgsl").to_string(),
        include_str!("shaders/includes/shadow_moments.wgsl").to_string(),
        include_str!("shaders/lights.wgsl").to_string(),
        include_str!("shaders/brdf/common.wgsl").to_string(),
        include_str!("shaders/brdf/lambert.wgsl").to_string(),
        include_str!("shaders/brdf/phong.wgsl").to_string(),
        include_str!("shaders/brdf/oren_nayar.wgsl").to_string(),
        include_str!("shaders/brdf/cook_torrance.wgsl").to_string(),
        include_str!("shaders/brdf/disney_principled.wgsl").to_string(),
        include_str!("shaders/brdf/ashikhmin_shirley.wgsl").to_string(),
        include_str!("shaders/brdf/ward.wgsl").to_string(),
        include_str!("shaders/brdf/toon.wgsl").to_string(),
        include_str!("shaders/brdf/minnaert.wgsl").to_string(),
        strip_includes(include_str!("shaders/brdf/dispatch.wgsl")),
        strip_includes(include_str!("shaders/lighting.wgsl")),
        include_str!("shaders/lighting_ibl.wgsl").to_string(),
        include_str!("shaders/terrain_noise.wgsl").to_string(),
        include_str!("shaders/terrain_probes.wgsl").to_string(),
        include_str!("shaders/includes/tonemap_common.wgsl").to_string(),
        strip_includes(include_str!("shaders/terrain_pbr_pom.wgsl")),
    ]
    .join("\n")
}

pub(crate) fn terrain_shadow_depth() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/terrain_shadow_depth.wgsl").to_string(),
    ]
    .join("\n")
}

#[cfg(any(test, all(feature = "enable-pbr", feature = "enable-tbn")))]
pub(crate) fn pbr() -> String {
    let shadows = crate::shadows::CsmRenderer::shader_source().replace("@group(2)", "@group(3)");
    [
        shadows,
        include_str!("shaders/lights.wgsl").to_string(),
        include_str!("shaders/brdf/common.wgsl").to_string(),
        include_str!("shaders/brdf/lambert.wgsl").to_string(),
        include_str!("shaders/brdf/phong.wgsl").to_string(),
        include_str!("shaders/brdf/oren_nayar.wgsl").to_string(),
        include_str!("shaders/brdf/cook_torrance.wgsl").to_string(),
        include_str!("shaders/brdf/disney_principled.wgsl").to_string(),
        include_str!("shaders/brdf/ashikhmin_shirley.wgsl").to_string(),
        include_str!("shaders/brdf/ward.wgsl").to_string(),
        include_str!("shaders/brdf/toon.wgsl").to_string(),
        include_str!("shaders/brdf/minnaert.wgsl").to_string(),
        strip_includes(include_str!("shaders/brdf/dispatch.wgsl")),
        strip_includes(include_str!("shaders/lighting.wgsl")),
        include_str!("shaders/lighting_ibl.wgsl").to_string(),
        include_str!("shaders/includes/tonemap_common.wgsl").to_string(),
        strip_includes(include_str!("shaders/pbr.wgsl")),
    ]
    .join("\n")
}

#[cfg(test)]
pub(crate) fn assert_valid_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
}

#[cfg(test)]
pub(crate) fn assert_valid_wgsl_without_gpu(label: &str, source: &str) {
    assert_valid_wgsl(source);
    eprintln!("{label}: live GPU unavailable; validated the WGSL contract statically");
}

/// Descriptor-indexing variant of [`terrain`]: the single virtual-texture atlas
/// becomes a `binding_array` indexed by the family slot. Selected only when the
/// adapter grants `TEXTURE_COMPRESSION_BC`, `TEXTURE_BINDING_ARRAY` and
/// non-uniform indexing; otherwise the compatibility assembly in [`terrain`] is
/// compiled and a `terrain_vt_bindless_atlas` degradation is recorded.
///
/// This is the last source-level substitution in the renderer: the two atlas
/// forms cannot be expressed as one WGSL declaration. `terrain_atlas_variants_are_distinct`
/// fails if either substitution silently stops matching.
/// Number of atlas textures in the bindless `binding_array`.
///
/// The WGSL array must be SIZED and must match the `count` on the bind-group
/// layout entry exactly. An unsized `binding_array` against a layout declaring
/// a count is a descriptor-array mismatch that Vulkan faults on at draw time
/// (the device is lost, with no validation error first); DX12 tolerated it.
/// `terrain/renderer/bind_groups/layouts.rs` reads this same constant.
pub(crate) const VT_ATLAS_BINDING_COUNT: u32 = 3;

pub(crate) fn terrain_bindless() -> String {
    terrain()
        .replace(
            "var terrain_vt_atlas: texture_2d<f32>;",
            &format!(
                "var terrain_vt_atlas: binding_array<texture_2d<f32>, {VT_ATLAS_BINDING_COUNT}>;"
            ),
        )
        .replace(
            "textureSampleLevel(terrain_vt_atlas, terrain_vt_sampler, atlas_uv, 0.0)",
            "textureSampleLevel(terrain_vt_atlas[terrain_vt_atlas_layer(family_slot)], terrain_vt_sampler, atlas_uv, 0.0)",
        )
}

fn terrain_base(bindless: bool) -> String {
    if bindless {
        terrain_bindless()
    } else {
        terrain()
    }
}

/// TESSELLA pass 1: the shared terrain module plus the visibility-write
/// fragment stage. Depth + R32Uint primitive identity only, no material work.
pub(crate) fn terrain_visbuffer_write(bindless: bool) -> String {
    [
        terrain_base(bindless),
        include_str!("shaders/terrain_visbuffer_write.wgsl").to_string(),
    ]
    .join("\n")
}

/// TESSELLA pass 2: the shared terrain module plus the full-screen material
/// resolve that decodes pass 1's identity and shades each visible pixel once.
pub(crate) fn terrain_visbuffer_resolve(bindless: bool) -> String {
    [
        terrain_base(bindless),
        include_str!("shaders/terrain_visibility_fullscreen.wgsl").to_string(),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every source the renderer can hand to naga, in the exact form the
    /// pipeline cache builds it. Nothing here is reassembled by the test: a
    /// variant that only exists inside a test proves nothing about the device.
    fn every_renderer_source() -> Vec<(&'static str, String)> {
        let stats = include_str!("shaders/terrain_visbuffer_resolve.wgsl").to_string();
        vec![
            ("hybrid_kernel", hybrid_kernel()),
            ("aether_sky", aether_sky()),
            ("aether_blit", aether_blit()),
            ("prometheus_aerial", prometheus_aerial()),
            ("terrain", terrain()),
            ("terrain_bindless", terrain_bindless()),
            ("visbuffer_write", terrain_visbuffer_write(false)),
            ("visbuffer_write_bindless", terrain_visbuffer_write(true)),
            ("visbuffer_resolve", terrain_visbuffer_resolve(false)),
            (
                "visbuffer_resolve_bindless",
                terrain_visbuffer_resolve(true),
            ),
            ("visbuffer_stats", stats),
        ]
    }

    #[test]
    fn assembled_renderer_sources_are_valid_wgsl() {
        for (name, source) in every_renderer_source() {
            assert_valid_wgsl(&source);
            assert!(!source.is_empty(), "{name} assembled an empty shader");
        }
    }

    #[test]
    fn production_aether_consumers_share_one_evaluation_core() {
        let marker = "AETHER's single production LUT-evaluation core";
        let core = include_str!("shaders/atmosphere/evaluation_core.wgsl");
        assert!(!core.contains("@group("));
        assert!(!core.contains("@binding("));
        for (name, source) in [
            ("sky", aether_sky()),
            ("terrain", terrain()),
            ("prometheus", prometheus_aerial()),
        ] {
            assert_eq!(
                source.matches(marker).count(),
                1,
                "{name} must assemble exactly one AETHER evaluation core"
            );
            assert!(source.contains("AETHER_EVAL_WAVELENGTHS_NM"));
            assert!(source.contains("AETHER_EVAL_CIE_XYZ"));
            assert!(source.contains("fn aether_eval_xyz_to_rgb"));
            assert!(source.contains("fn aether_eval_mu_to_unit"));
            assert!(source.contains("fn aether_eval_nu_to_unit"));
            assert!(source.contains("fn aether_eval_sample_accumulated_scattering"));
            assert!(source.contains("fn aether_eval_segment_transmittance"));
        }

        let sky = include_str!("shaders/atmosphere/scattering.wgsl");
        let terrain_source = include_str!("shaders/terrain_pbr_pom.wgsl");
        let prometheus = include_str!("shaders/atmosphere/prometheus_aerial.wgsl");
        assert!(sky.contains("aether_eval_sample_accumulated_scattering("));
        assert!(terrain_source.contains("aether_eval_sample_accumulated_scattering("));
        assert!(terrain_source.contains("aether_eval_segment_transmittance("));
        assert!(prometheus.contains("aether_eval_sample_accumulated_scattering("));
        assert!(prometheus.contains("aether_eval_segment_transmittance("));

        for duplicate in [
            "AETHER_TERRAIN_WAVELENGTHS_NM",
            "AETHER_TERRAIN_CIE_XYZ",
            "AETHER_TERRAIN_OZONE_ABSORPTION",
            "fn aether_terrain_xyz_to_rgb",
            "fn aether_terrain_spectral_xyz",
            "fn aether_terrain_mu_to_unit",
            "fn aether_terrain_nu_to_unit",
            "fn aether_terrain_load_scattering",
            "AETHER_PT_WAVELENGTHS_NM",
            "AETHER_PT_CIE_XYZ",
            "AETHER_PT_OZONE_ABSORPTION",
            "fn prometheus_xyz_to_rgb",
            "fn prometheus_mu_to_unit",
            "fn prometheus_nu_to_unit",
            "fn prometheus_segment_transmittance",
            "fn prometheus_load_scattering_texel",
            "fn atmosphere_mu_to_unit",
            "fn atmosphere_relative_cosine_to_unit",
            "fn atmosphere_load_scattering",
        ] {
            assert!(
                !sky.contains(duplicate)
                    && !terrain_source.contains(duplicate)
                    && !prometheus.contains(duplicate),
                "production consumer retained divergent evaluator {duplicate}"
            );
        }

        let stochastic = include_str!("shaders/atmosphere/prometheus_spectral_reference.wgsl");
        assert!(!stochastic.contains(marker));
        assert!(!stochastic.contains("aether_eval_sample_accumulated_scattering"));
    }

    #[test]
    fn terrain_height_sampling_is_portable_manual_bilinear() {
        let source = terrain();
        assert!(source.contains("fn sample_height_bilinear_level("));
        assert_eq!(source.matches("textureLoad(height_tex").count(), 4);
        assert!(!source.contains("textureSample(height_tex"));
        assert!(!source.contains("textureSampleLevel(height_tex"));

        // Both the ordinary geometry path and every clipmap morph lookup must
        // share the same reconstruction instead of drifting by callsite.
        assert!(source.contains("let h_raw = sample_height_bilinear(uv);"));
        assert!(source.contains("let h_fine = sample_height_bilinear(uv);"));
        assert_eq!(
            source.matches("sample_height_bilinear(coarse_base").count(),
            4
        );

        // Resolve reconstructs the exact vertices emitted by the visibility
        // write pass. It must therefore reuse the same helper rather than
        // silently reintroducing filterable R32Float sampling.
        let resolve = terrain_visbuffer_resolve(false);
        assert!(!resolve.contains("textureSample(height_tex"));
        assert!(!resolve.contains("textureSampleLevel(height_tex"));
        assert_eq!(resolve.matches("textureLoad(height_tex").count(), 4);
        assert!(resolve.contains("let h_fine = sample_height_bilinear(uv);"));

        // The CSM caster and visible surface must agree between texel centres.
        let shadow = terrain_shadow_depth();
        assert!(shadow.contains("fn sample_height_bilinear("));
        assert_eq!(shadow.matches("textureLoad(height_tex").count(), 4);
        assert!(!shadow.contains("textureSample(height_tex"));
        assert!(!shadow.contains("textureSampleLevel(height_tex"));
        assert!(shadow.contains("let h_raw = sample_height_bilinear(uv);"));
    }

    #[test]
    fn actual_pbr_terrain_and_csm_assemblies_are_valid_wgsl() {
        for source in [
            pbr(),
            terrain(),
            terrain_shadow_depth(),
            crate::shadows::CsmRenderer::shader_source().to_owned(),
        ] {
            assert_valid_wgsl(&source);
        }
    }

    /// The visibility packing is defined by real files now, not by rewriting
    /// the forward module's text at assembly time.
    #[test]
    fn visibility_passes_are_distinct_modules_with_one_packing() {
        let write = terrain_visbuffer_write(false);
        let resolve = terrain_visbuffer_resolve(false);
        assert!(write.contains("fn fs_visibility("));
        assert!(write.contains("fn terrain_visbuffer_pack("));
        assert!(!write.contains("fn fs_visibility_resolve_fullscreen("));
        assert!(resolve.contains("fn fs_visibility_resolve_fullscreen("));
        assert!(!resolve.contains("fn fs_visibility("));
        for bindless in [false, true] {
            assert_ne!(
                terrain_visbuffer_write(bindless),
                terrain_visbuffer_resolve(bindless),
                "the two visibility pipelines must not share one source"
            );
        }
        // The forward module carries the packed tile/LOD identity itself; the
        // stale 24|8 packing and its string rewrite are gone.
        //
        // The retired mask is FORMATTED rather than written as a literal: this
        // file is itself scanned by `test_visibility_write_and_resolve_are_
        // separate_shader_sources`, so a literal here would be an occurrence of
        // the very constant the gate requires to be absent from the assembly.
        let retired_primitive_mask = format!("0x{:08x}u", 0x00ff_ffffu32);
        let terrain = terrain();
        assert!(!terrain.contains("fn fs_visibility("));
        assert!(!terrain.contains(&retired_primitive_mask));
        assert!(terrain.contains("out.tile_id = ((_tile_id_lod.y & 0xfu) << 12u)"));
    }

    /// Guards the one remaining source substitution: if either atlas pattern
    /// stops matching, `terrain_bindless()` silently degenerates into the
    /// compatibility source and the descriptor-indexing path stops existing.
    #[test]
    fn terrain_atlas_variants_are_distinct() {
        let fixed = terrain();
        let bindless = terrain_bindless();
        assert!(fixed.contains("var terrain_vt_atlas: texture_2d<f32>;"));
        assert!(!fixed.contains("binding_array"));
        assert!(!fixed.contains("terrain_vt_atlas[terrain_vt_atlas_layer(family_slot)]"));
        // The array must be SIZED and must agree with the bind-group layout's
        // `count`; an unsized array against a counted layout entry is what
        // faulted the Vulkan device.
        assert!(bindless.contains(&format!(
            "var terrain_vt_atlas: binding_array<texture_2d<f32>, {VT_ATLAS_BINDING_COUNT}>;"
        )));
        assert!(!bindless.contains("binding_array<texture_2d<f32>>"));
        assert!(bindless
            .contains("terrain_vt_atlas[terrain_vt_atlas_layer(family_slot)], terrain_vt_sampler"));
        assert!(bindless.contains("fn terrain_vt_atlas_layer(family_slot: u32) -> u32"));
        assert_ne!(fixed, bindless);
    }
}
