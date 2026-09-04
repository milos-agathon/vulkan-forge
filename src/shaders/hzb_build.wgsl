// HZB (Hierarchical Z-Buffer) build shader
// Generates a configurable depth pyramid for accelerated occlusion queries (P5)

// ============================================================================
// Generic copy pass: Depth texture -> R32Float mip 0.
// ============================================================================

@group(0) @binding(0) var depth_in: texture_depth_2d;
@group(0) @binding(1) var hzb_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_in);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let depth = textureLoad(depth_in, gid.xy, 0);
    textureStore(hzb_out, gid.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
}

// Terrain allocates a half-sized pyramid. Fuse the copy with the first
// conservative MAX reduction and never materialize a full-resolution R32Float
// mip. Proportional bounds preserve odd edges.
@compute @workgroup_size(8, 8, 1)
fn cs_copy_max_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(hzb_out);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }

    let src_dims = textureDimensions(depth_in);
    let src_lo = gid.xy * src_dims / dst_dims;
    let src_hi = min(
        ((gid.xy + 1u) * src_dims + dst_dims - 1u) / dst_dims,
        src_dims,
    );
    var reduced = 0.0;
    for (var y = src_lo.y; y < src_hi.y; y++) {
        for (var x = src_lo.x; x < src_hi.x; x++) {
            reduced = max(reduced, textureLoad(depth_in, vec2<u32>(x, y), 0));
        }
    }
    textureStore(hzb_out, gid.xy, vec4<f32>(reduced, 0.0, 0.0, 0.0));
}

// ============================================================================
// Downsample pass: R32Float mip N -> R32Float mip N+1
// ============================================================================

struct DownsampleParams {
    reversed_z: u32,  // 1 if using reversed-Z, 0 otherwise
}

@group(0) @binding(0) var hzb_src: texture_2d<f32>;
@group(0) @binding(1) var hzb_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: DownsampleParams;

@compute @workgroup_size(8, 8, 1)
fn cs_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(hzb_dst);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }
    
    let src_dims = textureDimensions(hzb_src);
    let reversed = params.reversed_z != 0u;
    // Proportional source bounds include the trailing row/column when an odd
    // source dimension is halved (for example, 5 -> 2).
    let src_lo = gid.xy * src_dims / dst_dims;
    let src_hi = min(
        ((gid.xy + 1u) * src_dims + dst_dims - 1u) / dst_dims,
        src_dims,
    );
    var reduced = select(1.0, 0.0, reversed);
    for (var y = src_lo.y; y < src_hi.y; y++) {
        for (var x = src_lo.x; x < src_hi.x; x++) {
            let depth = textureLoad(hzb_src, vec2<u32>(x, y), 0).r;
            reduced = select(min(reduced, depth), max(reduced, depth), reversed);
        }
    }

    textureStore(hzb_dst, gid.xy, vec4<f32>(reduced, 0.0, 0.0, 0.0));
}
