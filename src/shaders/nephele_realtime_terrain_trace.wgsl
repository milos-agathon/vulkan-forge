// Batched realtime terrain queries for NEPHELE. The production PROMETHEUS
// terrain_trace implementation is assembled before this adapter.
struct NepheleTraceUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera: vec4<f32>,
    sun: vec4<f32>,
    sun_radiance: vec4<f32>,
    sigma_s: vec4<f32>,
    sigma_t: vec4<f32>,
    depth: vec4<f32>,
    grid: vec4<u32>,
    viewport: vec4<u32>,
    scattering: vec4<f32>,
    terrain_albedo: vec4<f32>,
    diffuse_ibl: vec4<f32>,
}

struct NepheleTerrainHit {
    point_t: vec4<f32>,
    normal_hit: vec4<f32>,
}

@group(3) @binding(8) var<uniform> nephele: NepheleTraceUniforms;
@group(3) @binding(9) var nephele_blue_noise: texture_2d<u32>;
@group(3) @binding(10) var nephele_extinction: texture_3d<f32>;
@group(3) @binding(11) var<storage, read_write> nephele_sun_hits: array<NepheleTerrainHit>;
@group(3) @binding(12) var<storage, read_write> nephele_phase_hits: array<NepheleTerrainHit>;

fn nephele_froxel_distance(u: f32) -> f32 {
    return nephele.depth.x * exp(nephele.depth.z * clamp(u, 0.0, 1.0));
}

fn nephele_froxel_world(id: vec3<u32>) -> vec3<f32> {
    let visible = nephele.grid.xy - vec2<u32>(2u);
    let uv = (vec2<f32>(id.xy) + 0.5 - vec2<f32>(1.0)) / vec2<f32>(visible);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far_h = nephele.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let direction = normalize(far_h.xyz / far_h.w - nephele.camera.xyz);
    return nephele.camera.xyz + direction * nephele_froxel_distance(
        (f32(id.z) + 0.5) / f32(nephele.grid.z)
    );
}

fn nephele_rank(id: vec3<u32>, salt: u32) -> f32 {
    let dimensions = vec2<u32>(textureDimensions(nephele_blue_noise));
    let p = (id.xy + vec2<u32>(id.z * 3u + salt, id.z * 5u + salt * 7u)) % dimensions;
    let rank = textureLoad(nephele_blue_noise, vec2<i32>(p), 0).r;
    return (f32(rank) + 0.5) / 64.0;
}

fn nephele_phase_direction(incident: vec3<f32>, id: vec3<u32>) -> vec3<f32> {
    let u1 = nephele_rank(id, 0u);
    let u2 = nephele_rank(id, 1u);
    var cos_theta = 1.0 - 2.0 * u1;
    if (nephele.scattering.w >= 0.5) {
        let g = nephele.scattering.z;
        if (abs(g) > 1e-4) {
            let ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
            cos_theta = clamp((1.0 + g * g - ratio * ratio) / (2.0 * g), -1.0, 1.0);
        }
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 6.283185307179586 * u2;
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(incident.y) < 0.999);
    let tangent = normalize(cross(helper, incident));
    let bitangent = cross(incident, tangent);
    return normalize(tangent * (cos(phi) * sin_theta) + bitangent * (sin(phi) * sin_theta) + incident * cos_theta);
}

fn nephele_store_hit(index: u32, sun: bool, hit: HybridHitResult) {
    let encoded = NepheleTerrainHit(vec4<f32>(hit.point, hit.t), vec4<f32>(hit.normal, f32(hit.hit)));
    if (sun) {
        nephele_sun_hits[index] = encoded;
    } else {
        nephele_phase_hits[index] = encoded;
    }
}

@compute @workgroup_size(4, 4, 4)
fn main_nephele_realtime_terrain_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= nephele.grid.xyz)) {
        return;
    }
    let index = id.x + nephele.grid.x * (id.y + nephele.grid.y * id.z);
    let canonical = textureLoad(nephele_extinction, vec3<i32>(id), 0);
    if (!(canonical.a > 0.0)) {
        var miss: HybridHitResult;
        miss.t = nephele.depth.y;
        miss.point = vec3<f32>(0.0);
        miss.normal = vec3<f32>(0.0);
        miss.material_id = 0u;
        miss.hit_type = 3u;
        miss.hit = 0u;
        miss._pad = vec2<u32>(0u);
        nephele_store_hit(index, true, miss);
        nephele_store_hit(index, false, miss);
        return;
    }
    let world = nephele_froxel_world(id);
    let incident = normalize(world - nephele.camera.xyz);
    // These are two distinct physical terms, not two quality samples.
    let sun_hit = terrain_trace(Ray(world, 1e-4, normalize(nephele.sun.xyz), nephele.depth.y), false, true);
    let continuation = nephele_phase_direction(incident, id);
    let phase_hit = terrain_trace(Ray(world, 1e-4, continuation, nephele.depth.y), false, true);
    nephele_store_hit(index, true, sun_hit);
    nephele_store_hit(index, false, phase_hit);
}
