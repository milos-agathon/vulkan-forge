// P2.3: GPU LOD selection compute shader with frustum culling.
//
// This compute shader performs per-tile frustum culling and LOD selection
// on the GPU, outputting a compact list of visible tiles with optimal LOD levels.
//
// Workgroup size: 64 threads (8x8 tile grid per workgroup)

struct LodSelectParams {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    frustum_planes: array<vec4<f32>, 6>,  // left, right, bottom, top, near, far
    lod_params: vec4<f32>,    // x=pixel_error_budget, y=viewport_height, z=fov_y, w=max_lod
    terrain_params: vec4<f32>, // x=tile_size, y=num_tiles, z=variant_count, w=first_instance
    height_params: vec4<f32>,  // x=min world height, y=max world height, z=frustum enabled
}

struct TileInfo {
    tile_id: u32,       // packed: lod(8) | x(12) | y(12)
    height_min: f32,
    bounds_min: vec2<f32>,
    bounds_max: vec2<f32>,
    distance: f32,
    selected_lod: u32,
    visible: u32,       // 0 = culled, 1 = visible
    height_max: f32,
}

struct OutputHeader {
    visible_count: atomic<u32>,
    total_triangles: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

struct IndirectDrawTemplate {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    tile_id: u32,
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

@group(0) @binding(0) var<uniform> params: LodSelectParams;
@group(0) @binding(1) var<storage, read> input_tiles: array<TileInfo>;
@group(0) @binding(2) var<storage, read_write> output_tiles: array<TileInfo>;
@group(0) @binding(3) var<storage, read_write> output_header: OutputHeader;
@group(0) @binding(4) var<storage, read> draw_templates: array<IndirectDrawTemplate>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<DrawIndexedIndirectArgs>;
@group(0) @binding(6) var<storage, read_write> draw_instances: array<ClipmapDrawInstance>;

// Pack tile ID from components
fn pack_tile_id(lod: u32, x: u32, y: u32) -> u32 {
    return (lod << 24u) | ((x & 0xFFFu) << 12u) | (y & 0xFFFu);
}

// Unpack tile ID to components
fn unpack_tile_id(packed: u32) -> vec3<u32> {
    let lod = packed >> 24u;
    let x = (packed >> 12u) & 0xFFFu;
    let y = packed & 0xFFFu;
    return vec3<u32>(lod, x, y);
}

// Test if a point is inside a plane (positive half-space)
fn point_in_plane(point: vec3<f32>, plane: vec4<f32>) -> bool {
    return dot(plane.xyz, point) + plane.w >= 0.0;
}

// Test if an AABB is visible against frustum planes
fn frustum_cull_aabb(bounds_min: vec2<f32>, bounds_max: vec2<f32>, height_min: f32, height_max: f32) -> bool {
    // Test each frustum plane
    for (var i = 0u; i < 6u; i++) {
        let plane = params.frustum_planes[i];
        
        // Find the positive vertex (furthest along plane normal)
        var p_vertex = vec3<f32>(bounds_min.x, bounds_min.y, height_min);
        if plane.x >= 0.0 { p_vertex.x = bounds_max.x; }
        if plane.y >= 0.0 { p_vertex.y = bounds_max.y; }
        if plane.z >= 0.0 { p_vertex.z = height_max; }
        
        // If positive vertex is outside plane, AABB is culled
        if !point_in_plane(p_vertex, plane) {
            return false;
        }
    }
    return true;
}

// Calculate projected geometric error for a candidate LOD.
fn calculate_screen_space_error(distance: f32, tile_size: f32, lod: u32) -> f32 {
    let viewport_height = params.lod_params.y;
    let fov_y = params.lod_params.z;
    
    // Avoid division by very small distances
    let safe_distance = max(distance, 0.1);
    
    let half_fov = fov_y * 0.5;
    let pixels_per_unit = (viewport_height * 0.5) / (safe_distance * tan(half_fov));
    let geometric_error = tile_size * f32(1u << lod);
    return geometric_error * pixels_per_unit;
}

// Select optimal LOD for a tile based on distance
fn select_lod(distance: f32, tile_size: f32) -> u32 {
    let max_lod = u32(params.lod_params.w);
    let pixel_error_budget = params.lod_params.x;
    
    // Choose the coarsest variant that still fits the pixel-error budget.
    for (var lod = i32(max_lod); lod >= 0; lod--) {
        let error = calculate_screen_space_error(distance, tile_size, u32(lod));
        if error <= pixel_error_budget {
            return u32(lod);
        }
    }
    return 0u;
}

@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x != 0u {
        return;
    }
    let num_tiles = u32(params.terrain_params.y);
    for (var tile_index = 0u; tile_index < num_tiles; tile_index++) {
        var tile = input_tiles[tile_index];
    
    // Calculate tile center and distance to camera
    let tile_center = (tile.bounds_min + tile.bounds_max) * 0.5;
    let camera_pos_2d = params.camera_pos.xy;
    let distance = length(tile_center - camera_pos_2d);
    tile.distance = distance;
    
    let has_local_heights = tile.height_min <= tile.height_max;
    let height_min = select(params.height_params.x, tile.height_min, has_local_heights);
    let height_max = select(params.height_params.y, tile.height_max, has_local_heights);
    let visible = params.height_params.z < 0.5 || frustum_cull_aabb(
            tile.bounds_min,
            tile.bounds_max,
            height_min,
            height_max,
        );
    tile.visible = select(0u, 1u, visible);
    
        if visible {
        // Select optimal LOD
        let tile_size = params.terrain_params.x;
        let selected_lod = select_lod(distance, tile_size);
        tile.selected_lod = selected_lod;
        
        // Append to output using atomic counter
        let output_idx = atomicAdd(&output_header.visible_count, 1u);
        output_tiles[output_idx] = tile;
        let variant_count = u32(params.terrain_params.z);
        let draw_template = draw_templates[tile_index * variant_count + selected_lod];
        let first_instance = select(0u, output_idx, params.terrain_params.w > 0.5);
        indirect_args[output_idx] = DrawIndexedIndirectArgs(
            draw_template.index_count,
            1u,
            draw_template.first_index,
            draw_template.base_vertex,
            first_instance,
        );
        draw_instances[output_idx] = ClipmapDrawInstance(
            mat4x4<f32>(
                vec4<f32>(1.0, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, 1.0, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 1.0, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0),
            ),
            vec2<u32>(draw_template.tile_id, selected_lod),
            vec2<u32>(0u),
        );
        
        // Accumulate triangle count
        let tri_count = draw_template.index_count / 3u;
            atomicAdd(&output_header.total_triangles, tri_count);
        }
    }
}

// Secondary pass: sort visible tiles by distance (optional, for front-to-back rendering)
@compute @workgroup_size(64, 1, 1)
fn cs_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Simple bubble sort pass - would be replaced with parallel sort for production
    let idx = global_id.x;
    let count = atomicLoad(&output_header.visible_count);
    
    if idx >= count - 1u {
        return;
    }
    
    // Compare adjacent tiles and swap if needed (distance-based)
    let tile_a = output_tiles[idx];
    let tile_b = output_tiles[idx + 1u];
    
    if tile_a.distance > tile_b.distance {
        output_tiles[idx] = tile_b;
        output_tiles[idx + 1u] = tile_a;
    }
}
