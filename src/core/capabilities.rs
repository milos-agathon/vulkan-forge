//! Negotiated GPU capability set: what forge3d wants, requires, and got.
use crate::core::degradation::record_degradation;
use wgpu::Features;

/// (stable name, feature bit, consequence when absent)
pub const WANTED: &[(&str, Features, &str)] = &[
    (
        "timestamp_query",
        Features::TIMESTAMP_QUERY,
        "per-pass GPU timings unavailable; certificate passes[].gpu_ms will be 0",
    ),
    (
        "pipeline_statistics_query",
        Features::PIPELINE_STATISTICS_QUERY,
        "pipeline_stats absent from certificate passes[]",
    ),
    (
        "texture_binding_array",
        Features::TEXTURE_BINDING_ARRAY,
        "terrain LUT texture-array bind path disabled; single-LUT rebind path used",
    ),
    (
        "sampled_texture_and_storage_buffer_array_non_uniform_indexing",
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        "non-uniform descriptor indexing unavailable; bindless paths disabled",
    ),
    (
        // The terrain visibility-buffer write entry uses
        // `@builtin(primitive_index)`. Vulkan gates that behind
        // SHADER_PRIMITIVE_INDEX; DX12 exposes SV_PrimitiveID unconditionally,
        // which is why omitting it faulted only on Vulkan.
        "shader_primitive_index",
        Features::SHADER_PRIMITIVE_INDEX,
        "@builtin(primitive_index) unavailable; the terrain visibility-buffer write pass cannot compile",
    ),
    (
        "texture_compression_bc",
        Features::TEXTURE_COMPRESSION_BC,
        "BC-compressed textures cannot upload natively; loader reports unsupported",
    ),
    (
        "float32_filterable",
        Features::FLOAT32_FILTERABLE,
        "float32 textures sample without filtering (nearest)",
    ),
    (
        "indirect_first_instance",
        Features::INDIRECT_FIRST_INSTANCE,
        "indirect draws with non-zero first_instance unavailable",
    ),
    (
        "multi_draw_indirect",
        Features::MULTI_DRAW_INDIRECT,
        "terrain indirect draws are issued in a CPU loop instead of one multi-draw call",
    ),
    (
        "multi_draw_indirect_count",
        Features::MULTI_DRAW_INDIRECT_COUNT,
        "terrain compacted indirect counts cannot drive one GPU multi-draw call; a CPU draw loop is used",
    ),
];

#[derive(Clone, Copy, Debug)]
pub struct CapabilitySet {
    pub wanted: Features,
    pub required: Features,
    pub granted: Features,
}

impl CapabilitySet {
    pub fn wants() -> Features {
        WANTED
            .iter()
            .fold(Features::empty(), |acc, (_, f, _)| acc | *f)
    }

    /// Intersect wants with adapter features. Nothing is hard-required, so
    /// this never fails; every want not granted records a degradation.
    pub fn negotiate(adapter_features: Features) -> Self {
        let wanted = Self::wants();
        let granted = wanted & adapter_features;
        for (name, feature, consequence) in WANTED {
            if !granted.contains(*feature) {
                record_degradation("capability_absent", name, consequence);
            }
        }
        CapabilitySet {
            wanted,
            required: Features::empty(),
            granted,
        }
    }

    pub fn wanted_names(&self) -> Vec<&'static str> {
        WANTED
            .iter()
            .filter(|(_, f, _)| self.wanted.contains(*f))
            .map(|(n, _, _)| *n)
            .collect()
    }
    pub fn granted_names(&self) -> Vec<&'static str> {
        WANTED
            .iter()
            .filter(|(_, f, _)| self.granted.contains(*f))
            .map(|(n, _, _)| *n)
            .collect()
    }

    /// Record a driver rejection of the negotiated request and downgrade to
    /// an empty optional-feature set for one retry.
    pub fn downgrade_after_request_failure(&mut self, error: &str) {
        record_degradation("capability_request_failed", "negotiated_features", error);
        for (name, feature, consequence) in WANTED {
            if self.granted.contains(*feature) {
                record_degradation("capability_absent", name, consequence);
            }
        }
        self.granted = Features::empty();
    }
}

/// TESSELLA: how the terrain clipmap may issue its compacted indirect draws on
/// this device. Both fields are a pure function of the GRANTED feature set.
///
/// Deciding and RECORDING are split so a caller only claims "the CPU draw loop
/// ran" on a frame that actually issues indirect draws. The render certificate
/// drains a render-LOCAL degradation capture, so the negotiation-time
/// `capability_absent` rows recorded by [`CapabilitySet::negotiate`] never reach
/// it — the fallback has to name itself from inside the render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectDrawMode {
    /// `INDIRECT_FIRST_INSTANCE` granted: indirect args may carry a non-zero
    /// `first_instance`, so one instance-buffer binding serves every draw.
    pub first_instance: bool,
    /// One `multi_draw_indexed_indirect_count` call replaces the CPU draw loop.
    pub multi_draw_count: bool,
}

impl IndirectDrawMode {
    /// The two bits the GPU multi-draw path needs beyond `first_instance`.
    fn multi_draw() -> Features {
        Features::MULTI_DRAW_INDIRECT | Features::MULTI_DRAW_INDIRECT_COUNT
    }

    /// Pure decision; records nothing, so it is safe to call before knowing
    /// whether this frame takes the indirect path at all.
    pub fn negotiate(granted: Features) -> Self {
        let first_instance = granted.contains(Features::INDIRECT_FIRST_INSTANCE);
        Self {
            first_instance,
            multi_draw_count: first_instance && granted.contains(Self::multi_draw()),
        }
    }

    /// Record every rendering fallback this mode implies. Call once per frame
    /// that actually issues indirect terrain draws.
    ///
    /// `capability_absent` names the missing bit; this names the DRAW PATH that
    /// ran. It is the only honest signal when `MULTI_DRAW_INDIRECT(_COUNT)` is
    /// granted yet unusable because `INDIRECT_FIRST_INSTANCE` is absent: in that
    /// case no `capability_absent` row exists for the multi-draw bits at all and
    /// the certificate's `granted` list actively advertises them while a CPU
    /// loop issues the draws.
    pub fn record_fallbacks(&self, granted: Features) {
        if !self.first_instance {
            record_degradation(
                "rendering_fallback",
                "terrain_indirect_first_instance",
                "INDIRECT_FIRST_INSTANCE absent; the clipmap instance buffer is rebound once \
                 per indirect draw instead of being indexed by first_instance",
            );
        }
        if !self.multi_draw_count {
            let cause = if granted.contains(Self::multi_draw()) {
                "MULTI_DRAW_INDIRECT(_COUNT) granted but INDIRECT_FIRST_INSTANCE absent"
            } else {
                "MULTI_DRAW_INDIRECT/MULTI_DRAW_INDIRECT_COUNT absent"
            };
            record_degradation(
                "rendering_fallback",
                "terrain_multi_draw_indirect",
                &format!(
                    "{cause}; terrain clipmap draws are issued as a CPU \
                     draw_indexed_indirect loop of up to max_draw_count calls instead of one \
                     GPU multi-draw"
                ),
            );
        }
    }
}

/// TESSELLA: the BC7/BC5 + `binding_array` atlas path the terrain virtual
/// texture wants. Single source of truth for that three-bit conjunction.
pub fn bindless_bc_atlas_supported(granted: Features) -> bool {
    granted.contains(Features::TEXTURE_COMPRESSION_BC)
        && granted.contains(Features::TEXTURE_BINDING_ARRAY)
        && granted.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
}

/// Record the atlas fallbacks implied by `granted`. Must be called on EVERY
/// frame that has a live VT runtime: `ensure_runtime` only rebuilds the runtime
/// on a config change, so recording at construction leaves frames 2..N silent
/// and the certified settled frame carries nothing.
pub fn record_bindless_bc_fallbacks(granted: Features) {
    if bindless_bc_atlas_supported(granted) {
        return;
    }
    if !granted.contains(Features::TEXTURE_COMPRESSION_BC) {
        record_degradation(
            "rendering_fallback",
            "terrain_vt_bc_atlas",
            "adapter lacks TEXTURE_COMPRESSION_BC; using raw RGBA8 atlas uploads",
        );
    }
    if !granted.contains(Features::TEXTURE_BINDING_ARRAY)
        || !granted
            .contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    {
        record_degradation(
            "rendering_fallback",
            "terrain_vt_bindless_atlas",
            "adapter lacks descriptor indexing; using the single-atlas compatibility path",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::degradation::{
        begin_degradation_capture, finish_degradation_capture, Degradation,
    };

    #[test]
    fn intersection_only_grants_adapter_features() {
        let caps =
            CapabilitySet::negotiate(Features::TIMESTAMP_QUERY | Features::DEPTH_CLIP_CONTROL);
        assert!(caps.granted.contains(Features::TIMESTAMP_QUERY));
        assert!(!caps.granted.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(!caps.granted.contains(Features::DEPTH_CLIP_CONTROL)); // never request unwanted
        assert_eq!(caps.wanted, CapabilitySet::wants());
    }
    #[test]
    fn empty_adapter_grants_nothing_and_degrades() {
        crate::core::degradation::clear_degradations();
        let caps = CapabilitySet::negotiate(Features::empty());
        assert!(caps.granted.is_empty());
        let degs = crate::core::degradation::degradations_snapshot();
        assert_eq!(degs.len(), WANTED.len());
        assert!(degs.iter().all(|d| d.kind == "capability_absent"));
        crate::core::degradation::clear_degradations();
    }

    /// Drive `f` inside a render-local capture. The thread-local CAPTURE dedups
    /// on `(kind, name)` independently of the process-global SINK, so each
    /// sub-case observes its own consequence string instead of whichever one a
    /// sibling test recorded first.
    fn capture<F: FnOnce()>(f: F) -> Vec<Degradation> {
        begin_degradation_capture();
        f();
        finish_degradation_capture()
    }

    fn pairs(entries: &[Degradation]) -> Vec<(&str, &str)> {
        entries
            .iter()
            .map(|entry| (entry.kind.as_str(), entry.name.as_str()))
            .collect()
    }

    fn consequence<'a>(entries: &'a [Degradation], name: &str) -> &'a str {
        entries
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no degradation named {name}: {entries:?}"))
            .consequence
            .as_str()
    }

    const MDI: Features = Features::MULTI_DRAW_INDIRECT;
    const MDIC: Features = Features::MULTI_DRAW_INDIRECT_COUNT;
    const IFI: Features = Features::INDIRECT_FIRST_INSTANCE;

    #[test]
    fn full_indirect_support_takes_gpu_multi_draw_and_records_nothing() {
        let granted = IFI | MDI | MDIC;
        let mode = IndirectDrawMode::negotiate(granted);
        assert!(mode.first_instance);
        assert!(mode.multi_draw_count);
        assert_eq!(pairs(&capture(|| mode.record_fallbacks(granted))), vec![]);
    }

    #[test]
    fn no_indirect_features_records_both_draw_fallbacks() {
        let mode = IndirectDrawMode::negotiate(Features::empty());
        assert!(!mode.first_instance);
        assert!(!mode.multi_draw_count);
        let degs = capture(|| mode.record_fallbacks(Features::empty()));
        assert_eq!(
            pairs(&degs),
            vec![
                ("rendering_fallback", "terrain_indirect_first_instance"),
                ("rendering_fallback", "terrain_multi_draw_indirect"),
            ]
        );
        let multi = consequence(&degs, "terrain_multi_draw_indirect");
        assert!(
            multi.contains("MULTI_DRAW_INDIRECT/MULTI_DRAW_INDIRECT_COUNT absent"),
            "{multi}"
        );
        assert!(multi.contains("CPU draw_indexed_indirect"), "{multi}");
    }

    /// The case `capability_absent` structurally CANNOT express: both multi-draw
    /// bits are GRANTED (so no `capability_absent` row exists for them, and the
    /// certificate's `granted` list advertises them) yet the CPU loop runs
    /// because `INDIRECT_FIRST_INSTANCE` is missing.
    #[test]
    fn granted_multi_draw_without_first_instance_still_records_the_cpu_loop() {
        let granted = MDI | MDIC;
        let mode = IndirectDrawMode::negotiate(granted);
        assert!(!mode.multi_draw_count);
        let degs = capture(|| mode.record_fallbacks(granted));
        assert_eq!(
            pairs(&degs),
            vec![
                ("rendering_fallback", "terrain_indirect_first_instance"),
                ("rendering_fallback", "terrain_multi_draw_indirect"),
            ]
        );
        let multi = consequence(&degs, "terrain_multi_draw_indirect");
        assert!(multi.contains("INDIRECT_FIRST_INSTANCE absent"), "{multi}");
    }

    #[test]
    fn first_instance_only_records_just_the_multi_draw_fallback() {
        let mode = IndirectDrawMode::negotiate(IFI);
        assert!(mode.first_instance);
        assert!(!mode.multi_draw_count);
        assert_eq!(
            pairs(&capture(|| mode.record_fallbacks(IFI))),
            vec![("rendering_fallback", "terrain_multi_draw_indirect")]
        );
    }

    #[test]
    fn one_missing_multi_draw_half_still_disables_the_gpu_multi_draw() {
        for granted in [IFI | MDI, IFI | MDIC] {
            let mode = IndirectDrawMode::negotiate(granted);
            assert!(!mode.multi_draw_count, "{granted:?}");
            assert_eq!(
                pairs(&capture(|| mode.record_fallbacks(granted))),
                vec![("rendering_fallback", "terrain_multi_draw_indirect")],
                "{granted:?}"
            );
        }
    }

    const BC: Features = Features::TEXTURE_COMPRESSION_BC;
    const TBA: Features = Features::TEXTURE_BINDING_ARRAY;
    const NUI: Features = Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

    #[test]
    fn full_bindless_bc_support_records_nothing() {
        assert!(bindless_bc_atlas_supported(BC | TBA | NUI));
        assert_eq!(
            pairs(&capture(|| record_bindless_bc_fallbacks(BC | TBA | NUI))),
            vec![]
        );
    }

    #[test]
    fn missing_bc_records_the_raw_rgba8_atlas_fallback() {
        let granted = TBA | NUI;
        assert!(!bindless_bc_atlas_supported(granted));
        assert_eq!(
            pairs(&capture(|| record_bindless_bc_fallbacks(granted))),
            vec![("rendering_fallback", "terrain_vt_bc_atlas")]
        );
    }

    #[test]
    fn missing_descriptor_indexing_records_the_single_atlas_fallback() {
        for granted in [BC | TBA, BC | NUI] {
            assert!(!bindless_bc_atlas_supported(granted), "{granted:?}");
            assert_eq!(
                pairs(&capture(|| record_bindless_bc_fallbacks(granted))),
                vec![("rendering_fallback", "terrain_vt_bindless_atlas")],
                "{granted:?}"
            );
        }
    }

    #[test]
    fn no_bc_and_no_indexing_records_both_atlas_fallbacks() {
        assert_eq!(
            pairs(&capture(|| record_bindless_bc_fallbacks(Features::empty()))),
            vec![
                ("rendering_fallback", "terrain_vt_bc_atlas"),
                ("rendering_fallback", "terrain_vt_bindless_atlas"),
            ]
        );
    }
}
