// NEPHELE's slow reference adapter. This entry adds only batched query I/O;
// the intersection call resolves to PROMETHEUS' production terrain_trace
// assembled before this file.
struct NepheleTerrainRay {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
}

struct NepheleTerrainHit {
    point_t: vec4<f32>,
    normal_hit: vec4<f32>,
}

@group(3) @binding(8) var<storage, read> nephele_terrain_rays: array<NepheleTerrainRay>;
@group(3) @binding(9) var<storage, read_write> nephele_terrain_hits: array<NepheleTerrainHit>;

@compute @workgroup_size(1, 1, 1)
fn main_nephele_terrain_trace_adapter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query = nephele_terrain_rays[gid.x];
    let ray = Ray(
        query.origin_tmin.xyz,
        query.origin_tmin.w,
        query.direction_tmax.xyz,
        query.direction_tmax.w,
    );
    let hit = terrain_trace(ray, false, true);
    nephele_terrain_hits[gid.x].point_t = vec4<f32>(hit.point, hit.t);
    nephele_terrain_hits[gid.x].normal_hit = vec4<f32>(hit.normal, f32(hit.hit));
}
