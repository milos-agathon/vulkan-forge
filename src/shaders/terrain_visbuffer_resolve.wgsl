// TESSELLA visibility resolve accounting. The material resolve itself is the
// full-screen `fs_visibility_resolve_fullscreen` pass in
// terrain_visibility_fullscreen.wgsl; this compute pass reads back its
// one-record-per-visible-pixel contract (visible/background pixel counts plus
// the frame counters that pass wrote) without sampling or compacting an
// overdraw stream.

struct VisibilityStats {
    visible_pixels: atomic<u32>,
    feedback_records: atomic<u32>,
    material_invocations: atomic<u32>,
    background_pixels: atomic<u32>,
    fallback_texels: atomic<u32>,
    forward_material_invocations: atomic<u32>,
}

@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> stats: VisibilityStats;
@group(0) @binding(2) var<storage, read> frame_counters: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(visibility_ids);
    if gid.x >= dimensions.x || gid.y >= dimensions.y {
        return;
    }
    if gid.x == 0u && gid.y == 0u {
        atomicStore(&stats.feedback_records, frame_counters[1]);
        atomicStore(&stats.material_invocations, frame_counters[0]);
        atomicStore(&stats.fallback_texels, frame_counters[2]);
        atomicStore(&stats.forward_material_invocations, frame_counters[3]);
    }
    let packed = textureLoad(visibility_ids, vec2<i32>(gid.xy), 0).r;
    if packed == 0u {
        atomicAdd(&stats.background_pixels, 1u);
        return;
    }
    atomicAdd(&stats.visible_pixels, 1u);
}
