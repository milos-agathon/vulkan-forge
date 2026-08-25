// TESSELLA pass 1 (visibility write). `shader_sources::terrain_visbuffer_write`
// appends this file to the shared terrain module, which supplies `VertexOutput`
// and the `vs_clipmap_main` vertex stage. Pass 1 does NO material work: no POM,
// no virtual-texture sampling, no feedback. It writes depth plus one Rg32Uint
// primitive identity.
//
// Identity (authoritative; mirrored by `terrain::renderer::visibility_buffer`
// and consumed by the runtime `fs_visibility_geometry` owner test as well as
// the static/debug `fs_visibility_resolve_fullscreen` reconstruction):
//   x = 1 + tile_lod_id, where tile_lod_id = (selected_lod << 14) | tile_index
//   y = the complete triangle index within the draw range
// Adding one only to x reserves vec2(0) for background without truncating y.

@fragment
fn fs_visibility(
    input: VertexOutput,
    @builtin(primitive_index) primitive_index: u32,
) -> @location(0) vec2<u32> {
    return vec2<u32>(input.tile_id + 1u, primitive_index);
}
