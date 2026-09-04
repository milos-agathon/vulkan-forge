// TESSELLA pass 1 (visibility write). `shader_sources::terrain_visbuffer_write`
// appends this file to the shared terrain module, which supplies `VertexOutput`
// and the `vs_clipmap_main` vertex stage. Pass 1 does NO material work: no POM,
// no virtual-texture sampling, no feedback. It writes depth plus one R32Uint
// primitive identity.
//
// Packing (authoritative; mirrored by `terrain::renderer::visibility_buffer`
// and decoded by `fs_visibility_resolve_fullscreen`):
//   bits 31..16  tile_lod_id = ((selected_lod & 0xf) << 12) | (tile_index & 0xfff),
//                written into `VertexOutput.tile_id` by `vs_clipmap_main`
//   bits 15..0   triangle index within the tile's draw range
// The whole payload is stored PLUS ONE so zero stays reserved for background.

const TERRAIN_VISBUFFER_TILE_SHIFT: u32 = 16u;
const TERRAIN_VISBUFFER_TILE_MASK: u32 = 0xffffu;
const TERRAIN_VISBUFFER_PRIMITIVE_MASK: u32 = 0xffffu;

fn terrain_visbuffer_pack(tile_lod_id: u32, primitive: u32) -> u32 {
    return ((tile_lod_id & TERRAIN_VISBUFFER_TILE_MASK) << TERRAIN_VISBUFFER_TILE_SHIFT)
        | (primitive & TERRAIN_VISBUFFER_PRIMITIVE_MASK);
}

@fragment
fn fs_visibility(
    input: VertexOutput,
    @builtin(primitive_index) primitive_index: u32,
) -> @location(0) u32 {
    return terrain_visbuffer_pack(input.tile_id, primitive_index) + 1u;
}
