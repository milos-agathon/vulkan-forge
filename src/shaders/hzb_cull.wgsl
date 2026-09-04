// TESSELLA two-phase occlusion. Phase 1 uses the previous frame's conservative
// MAX-reduced HZB. Phase 2 uses the fresh phase-1 HZB and rejects only when
// every covered depth sample is closer.

struct CullParams {
    view_proj: mat4x4<f32>,
    height_bounds: vec4<f32>,
    viewport: vec4<f32>, // width, height, max_mip, hzb_valid
    control: vec4<u32>,  // phase, indirect first-instance, full-resolution HZB mip bias, padding
}

struct OutputHeader {
    visible_count: atomic<u32>,
    total_triangles: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

struct CullHeader {
    draw_count: atomic<u32>,
    rejected_count: atomic<u32>,
    projection_bypassed: atomic<u32>,
    background_bypassed: atomic<u32>,
}

struct TileInfo {
    tile_id: u32,
    height_min: f32,
    bounds_min: vec2<f32>,
    bounds_max: vec2<f32>,
    distance: f32,
    selected_lod: u32,
    visible: u32,
    height_max: f32,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct ClipmapDrawInstance {
    transform: mat4x4<f32>,
    tile_id_lod: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: CullParams;
@group(0) @binding(1) var hzb: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> input_header: OutputHeader;
@group(0) @binding(3) var<storage, read> input_tiles: array<TileInfo>;
@group(0) @binding(4) var<storage, read> input_args: array<DrawIndexedIndirectArgs>;
@group(0) @binding(5) var<storage, read> input_instances: array<ClipmapDrawInstance>;
@group(0) @binding(6) var<storage, read_write> output_args: array<DrawIndexedIndirectArgs>;
@group(0) @binding(7) var<storage, read_write> output_instances: array<ClipmapDrawInstance>;
@group(0) @binding(8) var<storage, read_write> rejected_indices: array<u32>;
@group(0) @binding(9) var<storage, read_write> output_header: CullHeader;

fn tile_occluded(tile: TileInfo) -> bool {
    if params.viewport.w < 0.5 {
        atomicAdd(&output_header.projection_bypassed, 1u);
        return false;
    }

    var uv_min = vec2<f32>(1.0);
    var uv_max = vec2<f32>(0.0);
    var nearest_depth = 1.0;
    let has_local_heights = tile.height_min <= tile.height_max;
    let height_min = select(params.height_bounds.x, tile.height_min, has_local_heights);
    let height_max = select(params.height_bounds.y, tile.height_max, has_local_heights);
    for (var z_index = 0u; z_index < 2u; z_index++) {
        let world_z = select(height_min, height_max, z_index != 0u);
        for (var y_index = 0u; y_index < 2u; y_index++) {
            let world_y = select(tile.bounds_min.y, tile.bounds_max.y, y_index != 0u);
            for (var x_index = 0u; x_index < 2u; x_index++) {
                let world_x = select(tile.bounds_min.x, tile.bounds_max.x, x_index != 0u);
                let clip = params.view_proj * vec4<f32>(world_x, world_y, world_z, 1.0);
                if clip.w <= 0.0 {
                    atomicAdd(&output_header.projection_bypassed, 1u);
                    return false;
                }
                let ndc = clip.xyz / clip.w;
                let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
                uv_min = min(uv_min, uv);
                uv_max = max(uv_max, uv);
                nearest_depth = min(nearest_depth, ndc.z);
            }
        }
    }

    uv_min = clamp(uv_min, vec2<f32>(0.0), vec2<f32>(1.0));
    uv_max = clamp(uv_max, vec2<f32>(0.0), vec2<f32>(1.0));
    if any(uv_min >= uv_max) || nearest_depth <= 0.0 {
        atomicAdd(&output_header.projection_bypassed, 1u);
        return false;
    }

    let pixel_extent = max(
        (uv_max.x - uv_min.x) * params.viewport.x,
        (uv_max.y - uv_min.y) * params.viewport.y,
    );
    let phase = params.control.x;
    let requested_mip = u32(max(floor(log2(max(pixel_extent, 1.0))) - 4.0, 0.0));
    // Terrain's half-sized pyramid mip 0 represents full-resolution mip 1.
    // Saturating subtraction keeps tiny footprints on that conservative base;
    // larger footprints select the same physical resolution as before.
    let relative_mip = requested_mip - min(requested_mip, params.control.z);
    let mip = min(relative_mip, u32(params.viewport.z));
    let dims = textureDimensions(hzb, i32(mip));
    let lo = min(vec2<u32>(uv_min * vec2<f32>(dims)), dims - 1u);
    let hi = min(
        vec2<u32>(max(ceil(uv_max * vec2<f32>(dims)), vec2<f32>(1.0))) - 1u,
        dims - 1u,
    );
    var farthest_occluder = 0.0;
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let depth = textureLoad(hzb, vec2<u32>(x, y), i32(mip)).r;
            farthest_occluder = max(farthest_occluder, depth);
        }
    }
    // Both phases take the farthest sample from a MAX-reduced pyramid, so the
    // operation remains conservative.
    let occluder = farthest_occluder;
    if occluder >= 0.999999 {
        atomicAdd(&output_header.background_bypassed, 1u);
        return false;
    }
    return nearest_depth > occluder + 0.00001;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let phase = params.control.x;
    let source_count = select(
        atomicLoad(&input_header.visible_count),
        atomicLoad(&input_header.total_triangles),
        phase == 2u,
    );
    let i = gid.x;
    if i >= source_count {
        return;
    }
    let source_index = select(i, rejected_indices[i], phase == 2u);
    let occluded = tile_occluded(input_tiles[source_index]);
    if !occluded {
        let out_index = atomicAdd(&output_header.draw_count, 1u);
        var draw = input_args[source_index];
        draw.first_instance = select(0u, out_index, params.control.y != 0u);
        output_args[out_index] = draw;
        output_instances[out_index] = input_instances[source_index];
    } else if phase == 1u {
        let rejected_index = atomicAdd(&output_header.rejected_count, 1u);
        rejected_indices[rejected_index] = source_index;
    }
}
