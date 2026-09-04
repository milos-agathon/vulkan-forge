// TESSELLA pass 2 visibility resolve module. `shader_sources::terrain_visbuffer_resolve`
// appends this file to the shared terrain module, which supplies VertexOutput,
// FragmentOutput and `shade_main`. The runtime depth-equal geometry entry point
// replays the R32Uint identity written by terrain_visbuffer_write.wgsl (which
// defines the packing); the fullscreen reconstruction below remains a static
// contract/debug path.

@group(7) @binding(0)
var terrain_visibility_ids: texture_2d<u32>;

@group(7) @binding(1)
var terrain_visibility_depth: texture_depth_2d;

struct VisibilityClipmapVertex {
    position: vec2<f32>,
    uv: vec2<f32>,
    morph_data: vec2<f32>,
}

struct VisibilityDrawTemplate {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    tile_id: u32,
}

struct VisibilityResolveMeta {
    variant_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(7) @binding(2)
var<storage, read> terrain_visibility_vertices: array<VisibilityClipmapVertex>;

@group(7) @binding(3)
var<storage, read> terrain_visibility_indices: array<u32>;

@group(7) @binding(4)
var<storage, read> terrain_visibility_templates: array<VisibilityDrawTemplate>;

@group(7) @binding(5)
var<uniform> terrain_visibility_meta: VisibilityResolveMeta;

struct VisibilityFullscreenOutput {
    @builtin(position) clip_position: vec4<f32>,
}

struct VisibilityReconstructedVertex {
    clip: vec4<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
}

fn visibility_reconstruct_vertex(index: u32) -> VisibilityReconstructedVertex {
    let source = terrain_visibility_vertices[index];
    let uv = clamp(source.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let h_fine = sample_height_bilinear(uv);
    // Geomorphing, mirrored from `vs_clipmap_main`. Reconstructing from the
    // FINE height alone put every morphing vertex at a different clip position
    // than pass 1 rasterised it from, which tilts the triangle and biases the
    // barycentrics -- and therefore the interpolated UV -- by a fraction of a
    // pixel across the whole morph band of every clipmap ring.
    let height_dims = vec2<f32>(textureDimensions(height_tex));
    let coarse_texels = exp2(min(max(source.morph_data.y, 0.0) + 1.0, 16.0));
    let coarse_step = vec2<f32>(coarse_texels)
        / max(height_dims - vec2<f32>(1.0), vec2<f32>(1.0));
    let coarse_cell = uv / coarse_step;
    let coarse_base = floor(coarse_cell) * coarse_step;
    let coarse_t = fract(coarse_cell);
    let h00 = sample_height_bilinear(clamp(coarse_base, vec2<f32>(0.0), vec2<f32>(1.0)));
    let h10 = sample_height_bilinear(clamp(coarse_base + vec2<f32>(coarse_step.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)));
    let h01 = sample_height_bilinear(clamp(coarse_base + vec2<f32>(0.0, coarse_step.y), vec2<f32>(0.0), vec2<f32>(1.0)));
    let h11 = sample_height_bilinear(clamp(coarse_base + coarse_step, vec2<f32>(0.0), vec2<f32>(1.0)));
    let h_coarse = mix(mix(h00, h10, coarse_t.x), mix(h01, h11, coarse_t.x), coarse_t.y);
    let h_raw = mix(h_fine, h_coarse, clamp(source.morph_data.x, 0.0, 1.0));
    let t_geom = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_disp = det_fma(apply_height_curve01(t_geom), h_max - h_min, h_min);
    let h_exag = u_terrain.spacing_h_exag.z;
    let h_center = (h_min + h_max) * 0.5;
    let skirt_offset = select(
        0.0,
        u_terrain.camera_mode_params.y * 0.001,
        source.morph_data.x < 0.0,
    );
    let centered = vec3<f32>(
        source.position,
        (h_disp - h_center - skirt_offset) * h_exag,
    );
    var out: VisibilityReconstructedVertex;
    out.clip = det_mat4_mul_vec4(
        u_terrain.proj,
        det_mat4_mul_vec4(
            u_terrain.view,
            vec4<f32>(centered, 1.0),
        ),
    );
    out.world = vec3<f32>(
        source.position,
        (h_disp - skirt_offset) * h_exag,
    );
    out.uv = uv;
    return out;
}

// A visible triangle's vertices sit in front of the near plane, so `w` is
// positive in practice. The guard removes the divide-by-zero NaN that nothing
// upstream rules out statically, matching how `visibility_barycentrics` guards
// its own denominator below.
fn visibility_safe_w(w: f32) -> f32 {
    return select(1e-6, w, abs(w) > 1e-6);
}

fn visibility_barycentrics(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let denominator = v0.x * v1.y - v1.x * v0.y;
    let inverse_denominator = 1.0 / select(
        1e-8,
        denominator,
        abs(denominator) > 1e-8,
    );
    let y = (v2.x * v1.y - v1.x * v2.y) * inverse_denominator;
    let z = (v0.x * v2.y - v2.x * v0.y) * inverse_denominator;
    return vec3<f32>(1.0 - y - z, y, z);
}

struct VisibilitySurfaceSample {
    world: vec3<f32>,
    uv: vec2<f32>,
}

// The rasteriser does not interpolate from the floating-point clip positions
// the vertex stage emitted. It converts them to framebuffer coordinates and
// SNAPS each one to a fixed-point sub-pixel grid before building the edge
// functions, so the barycentrics it hands the fragment stage belong to a
// slightly different triangle than the fp32 one. Vulkan guarantees at least
// `subPixelPrecisionBits = 4`; the reference RTX 3070 (and every desktop
// D3D12/Vulkan rasteriser this path runs on) reports 8, i.e. a 1/256 px grid.
// Reconstructing from unsnapped positions leaves a residual that is invisible
// on smooth gradients but flips any hard material threshold the surface
// crosses, which is exactly where the forward/visibility mismatch used to
// concentrate.
const TERRAIN_VISBUFFER_SUBPIXEL: f32 = 256.0;

fn visibility_snap_to_subpixel_grid(framebuffer_xy: vec2<f32>) -> vec2<f32> {
    return round(framebuffer_xy * TERRAIN_VISBUFFER_SUBPIXEL)
        / TERRAIN_VISBUFFER_SUBPIXEL;
}

// Clip space -> framebuffer pixel coordinates, matching the viewport transform
// the rasteriser applies ((0.5, 0.5) is the centre of the top-left pixel).
fn visibility_framebuffer_of(clip_xy: vec2<f32>, w: f32, dimensions: vec2<f32>) -> vec2<f32> {
    let ndc = clip_xy / w;
    return vec2<f32>(
        (ndc.x * 0.5 + 0.5) * dimensions.x,
        (0.5 - ndc.y * 0.5) * dimensions.y,
    );
}

// Perspective-correct interpolation of one reconstructed triangle at an
// arbitrary framebuffer position. Evaluating it OUTSIDE the triangle is
// deliberate: that extrapolation is what the rasteriser hands its helper lanes,
// and it is how the quad-aligned analytic gradients below are formed.
//
// Barycentric coordinates are invariant under the affine NDC -> framebuffer
// map, so working in pixel space costs nothing and is the only space in which
// the sub-pixel snap above is expressible.
fn visibility_sample_surface(
    framebuffer_xy: vec2<f32>,
    v0: VisibilityReconstructedVertex,
    v1: VisibilityReconstructedVertex,
    v2: VisibilityReconstructedVertex,
    w: vec3<f32>,
    dimensions: vec2<f32>,
) -> VisibilitySurfaceSample {
    let bary_screen = visibility_barycentrics(
        framebuffer_xy,
        visibility_snap_to_subpixel_grid(
            visibility_framebuffer_of(v0.clip.xy, w.x, dimensions)),
        visibility_snap_to_subpixel_grid(
            visibility_framebuffer_of(v1.clip.xy, w.y, dimensions)),
        visibility_snap_to_subpixel_grid(
            visibility_framebuffer_of(v2.clip.xy, w.z, dimensions)),
    );
    let perspective = bary_screen / w;
    let bary = perspective / max(
        perspective.x + perspective.y + perspective.z,
        1e-8,
    );
    var sample: VisibilitySurfaceSample;
    sample.world = v0.world * bary.x + v1.world * bary.y + v2.world * bary.z;
    sample.uv = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;
    return sample;
}

// The packed pass-1 identity at `pixel`, or 0 when the pixel is background.
// Out-of-range coordinates (an odd-sized target's last quad) clamp to the edge,
// which can only ever return a real neighbour's identity or 0.
fn visibility_pixel_identity(pixel: vec2<i32>) -> u32 {
    let limit = vec2<i32>(textureDimensions(terrain_visibility_ids)) - vec2<i32>(1);
    let clamped = clamp(pixel, vec2<i32>(0), limit);
    let encoded = textureLoad(terrain_visibility_ids, clamped, 0).x;
    let depth = textureLoad(terrain_visibility_depth, clamped, 0);
    return select(encoded, 0u, depth >= 1.0);
}

@vertex
fn vs_visibility_fullscreen(
    @builtin(vertex_index) vertex_index: u32,
) -> VisibilityFullscreenOutput {
    var out: VisibilityFullscreenOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_visibility_resolve_fullscreen(
    input: VisibilityFullscreenOutput,
) -> FragmentOutput {
    let pixel = vec2<i32>(input.clip_position.xy);
    let covered = visibility_pixel_identity(pixel);
    // Helper-lane emulation. The forward rasteriser runs a partially covered
    // 2x2 quad with EVERY lane carrying the covering triangle's attributes --
    // the uncovered lanes are helper invocations whose only job is to give the
    // covered lanes a correct `dpdx`/`dpdy`. A full-screen resolve pass has no
    // such lanes, so an uncovered pixel used to fall through with a garbage
    // identity (`0u - 1u`) and poison the quad derivative of every covered
    // neighbour: LOD selection, the LOD-aware height normal, the virtual-texture
    // mip and the Toksvig/edge normal gradients all read it. Adopting a covered
    // quad partner's triangle and extrapolating it to this pixel's own centre
    // reproduces exactly what the rasteriser would have interpolated here.
    let quad_base = pixel & vec2<i32>(-2, -2);
    var identity = covered;
    if (identity == 0u) {
        for (var lane = 0u; lane < 4u; lane = lane + 1u) {
            let partner = visibility_pixel_identity(
                quad_base + vec2<i32>(i32(lane & 1u), i32(lane >> 1u)),
            );
            if (identity == 0u) {
                identity = partner;
            }
        }
    }
    if (identity == 0u) {
        // Every lane of this quad is background, so no derivative depends on
        // it. All four lanes take this branch together: quad uniformity, and
        // therefore derivative uniformity, is preserved.
        discard;
    }
    let encoded = identity;
    let dimensions = vec2<f32>(textureDimensions(terrain_visibility_ids));
    // `@builtin(position).xy` is ALREADY the pixel centre in framebuffer space
    // ((0.5, 0.5) is the centre of the top-left pixel), so adding another half
    // texel shifted every reconstructed attribute one half pixel down-right.
    let payload = encoded - 1u;
    let tile_lod_id = payload >> 16u;
    let primitive = payload & 0xffffu;
    let selected_lod = (tile_lod_id >> 12u) & 0xfu;
    let tile_index = tile_lod_id & 0xfffu;
    let draw_template = terrain_visibility_templates[
        tile_index * terrain_visibility_meta.variant_count + selected_lod
    ];
    let first = draw_template.first_index + primitive * 3u;
    let v0 = visibility_reconstruct_vertex(
        u32(i32(terrain_visibility_indices[first]) + draw_template.base_vertex),
    );
    let v1 = visibility_reconstruct_vertex(
        u32(i32(terrain_visibility_indices[first + 1u]) + draw_template.base_vertex),
    );
    let v2 = visibility_reconstruct_vertex(
        u32(i32(terrain_visibility_indices[first + 2u]) + draw_template.base_vertex),
    );
    let w = vec3<f32>(
        visibility_safe_w(v0.clip.w),
        visibility_safe_w(v1.clip.w),
        visibility_safe_w(v2.clip.w),
    );
    let here = visibility_sample_surface(
        input.clip_position.xy, v0, v1, v2, w, dimensions);

    // Quad-aligned analytic gradients of THIS pixel's covering triangle. A
    // coarse derivative is evaluated once per 2x2 quad from its top-left lane,
    // so sampling the same triangle at the quad's three anchor centres
    // reproduces the value the rasteriser would have produced for it, whatever
    // the neighbouring pixels' triangles happen to be.
    let quad_origin = vec2<f32>(quad_base) + vec2<f32>(0.5, 0.5);
    let anchor00 = visibility_sample_surface(
        quad_origin, v0, v1, v2, w, dimensions);
    let anchor10 = visibility_sample_surface(
        quad_origin + vec2<f32>(1.0, 0.0), v0, v1, v2, w, dimensions);
    let anchor01 = visibility_sample_surface(
        quad_origin + vec2<f32>(0.0, 1.0), v0, v1, v2, w, dimensions);
    let feedback_ddx_uv = anchor10.uv - anchor00.uv;
    let feedback_ddy_uv = anchor01.uv - anchor00.uv;
    let feedback_ddx_world = anchor10.world - anchor00.world;
    let feedback_ddy_world = anchor01.world - anchor00.world;
    terrain_explicit_ddx_uv = feedback_ddx_uv;
    terrain_explicit_ddy_uv = feedback_ddy_uv;
    terrain_explicit_ddx_world = feedback_ddx_world;
    terrain_explicit_ddy_world = feedback_ddy_world;
    terrain_explicit_gradients = 1u;

    var surface: VertexOutput;
    surface.clip_position = input.clip_position;
    surface.world_position = here.world;
    surface.world_normal = vec3<f32>(0.0, 0.0, 1.0);
    surface.tex_coord = here.uv;
    surface.tile_id = tile_lod_id;
    let out = shade_main(surface);
    if (covered == 0u) {
        // An emulated helper lane. It has now contributed its derivatives to
        // the covered lanes of this quad and must contribute nothing else: no
        // colour, no material-invocation count and no feedback record. That is
        // what keeps `visibility_feedback_records == visible_pixels` exact.
        discard;
    } else {
        atomicAdd(&terrain_frame_counters.material_invocations, 1u);
        terrain_vt_write_surface_feedback(
            surface.tex_coord,
            surface.world_position,
            feedback_ddx_uv,
            feedback_ddy_uv,
            feedback_ddx_world,
            feedback_ddy_world,
            0u,
            input.clip_position.xy,
        );
        if (terrain_vt_uniforms.config2.w != 0u) {
            if (terrain_vt_enabled()
                && terrain_vt_family_enabled(TERRAIN_VT_FAMILY_ALBEDO)
                && out.source_id == 0u) {
                atomicAdd(&terrain_frame_counters.fallback_texels, 1u);
            }
        }
    }
    return out;
}

// Depth-equal geometry resolve. Replaying the exact pass-1 clipmap vertex
// path preserves the fixed-function interpolants and quad derivatives that a
// full-screen reconstruction cannot reproduce bit-for-bit. The visibility ID
// check keeps overlapping/equal-depth geometry from paying the material cost
// more than once.
@fragment
fn fs_visibility_geometry(
    input: VertexOutput,
    @builtin(primitive_index) primitive_index: u32,
) -> FragmentOutput {
    // Capture the rasteriser's real quad gradients before the visibility-ID
    // ownership test discards neighbouring helper/non-owning lanes. The
    // Grid-UV normal and mask families now consume UV derivatives, while
    // albedo consumes world-space triplanar derivatives; evaluating either
    // after the divergent discard makes boundary derivatives undefined and
    // produced a one-pixel forward/visibility mismatch on Metal.
    let resolve_ddx_uv = dpdx(input.tex_coord);
    let resolve_ddy_uv = dpdy(input.tex_coord);
    let resolve_ddx_world = dpdx(input.world_position);
    let resolve_ddy_world = dpdy(input.world_position);
    let pixel = vec2<i32>(input.clip_position.xy);
    let encoded = textureLoad(terrain_visibility_ids, pixel, 0).x;
    let expected = (((input.tile_id & 0xffffu) << 16u)
        | (primitive_index & 0xffffu)) + 1u;
    if (encoded != expected) {
        discard;
    }
    terrain_explicit_ddx_uv = resolve_ddx_uv;
    terrain_explicit_ddy_uv = resolve_ddy_uv;
    terrain_explicit_ddx_world = resolve_ddx_world;
    terrain_explicit_ddy_world = resolve_ddy_world;
    terrain_explicit_gradients = 1u;
    let out = shade_main(input);
    atomicAdd(&terrain_frame_counters.material_invocations, 1u);
    terrain_vt_write_surface_feedback(
        input.tex_coord,
        input.world_position,
        resolve_ddx_uv,
        resolve_ddy_uv,
        resolve_ddx_world,
        resolve_ddy_world,
        0u,
        input.clip_position.xy,
    );
    if (terrain_vt_uniforms.config2.w != 0u) {
        if (terrain_vt_enabled()
            && terrain_vt_family_enabled(TERRAIN_VT_FAMILY_ALBEDO)
            && out.source_id == 0u) {
            atomicAdd(&terrain_frame_counters.fallback_texels, 1u);
        }
    }
    return out;
}
