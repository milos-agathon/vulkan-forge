// src/shaders/hybrid_traversal.wgsl
// Hybrid traversal combining mesh BVH traversal with legacy hooks for SDF.

// GPU hybrid traversal currently focuses on mesh geometry; SDF support is CPU-only.

// Hybrid traversal configuration (GPU path currently supports mesh traversal only)

// Hybrid scene data structures
struct HybridUniforms {
    sdf_primitive_count: u32,
    sdf_node_count: u32,
    mesh_vertex_count: u32,
    mesh_index_count: u32,
    mesh_bvh_node_count: u32,
    traversal_mode: u32, // 0 = hybrid, 1 = SDF only, 2 = mesh only, 3 = terrain only
    _pad: vec2u,
}

struct HybridHitResult {
    t: f32,
    point: vec3f,
    normal: vec3f,
    material_id: u32,
    hit_type: u32, // 0 = mesh, retained for compatibility
    hit: u32, // 0 = false, 1 = true
    _pad: vec2u,
}

struct Ray {
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
}

// BVH structures (matching existing pt_kernel.wgsl)
struct BvhNode {
    aabb_min: vec3f,
    left: u32,
    aabb_max: vec3f,
    right: u32,
    flags: u32,
    _pad: u32,
}

struct MeshVertex {
    position: vec3f,
    _pad: f32,
}

// Bind groups for hybrid traversal
// NOTE: Consolidated into group(1) to stay within max_bind_groups=4
@group(1) @binding(1) var<uniform> hybrid_uniforms: HybridUniforms;
@group(1) @binding(2) var<storage, read> mesh_vertices: array<MeshVertex>;
@group(1) @binding(3) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(4) var<storage, read> mesh_bvh_nodes: array<BvhNode>;

// Ray-AABB intersection for BVH traversal
fn ray_aabb_intersect(ray: Ray, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    var tmin = ray.tmin;
    var tmax = ray.tmax;

    for (var i = 0u; i < 3u; i = i + 1u) {
        let inv_dir = 1.0 / ray.direction[i];
        var t0 = (aabb_min[i] - ray.origin[i]) * inv_dir;
        var t1 = (aabb_max[i] - ray.origin[i]) * inv_dir;

        if (inv_dir < 0.0) {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tmin = max(tmin, t0);
        tmax = min(tmax, t1);

        if (tmin > tmax) {
            return false;
        }
    }

    return true;
}

// Ray-triangle intersection
fn ray_triangle_intersect(
    ray: Ray,
    v0: vec3f,
    v1: vec3f,
    v2: vec3f
) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);

    let epsilon = 1e-7;
    if (abs(a) < epsilon) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t > ray.tmin && t < ray.tmax) {
        let normal = normalize(cross(edge1, edge2));
        result.hit = 1u;
        result.t = t;
        result.point = ray.origin + ray.direction * t;
        result.normal = normal;
        result.material_id = 0u; // Default mesh material
        result.hit_type = 0u; // mesh
    }

    return result;
}

// BVH traversal for mesh intersection
const MAX_BVH_STACK_SIZE: u32 = 32u;

fn intersect_mesh(ray: Ray) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

    // Brute-force triangle sweep. This keeps the shader simple and guarantees
    // we shade meshes even when no GPU BVH data is available.
    let index_count = hybrid_uniforms.mesh_index_count;
    if (index_count < 3u) {
        return result;
    }

    for (var tri = 0u; tri + 2u < index_count; tri = tri + 3u) {
        let i0 = mesh_indices[tri];
        let i1 = mesh_indices[tri + 1u];
        let i2 = mesh_indices[tri + 2u];

        if (i0 >= hybrid_uniforms.mesh_vertex_count ||
            i1 >= hybrid_uniforms.mesh_vertex_count ||
            i2 >= hybrid_uniforms.mesh_vertex_count) {
            continue;
        }

        let v0 = mesh_vertices[i0].position;
        let v1 = mesh_vertices[i1].position;
        let v2 = mesh_vertices[i2].position;

        let tri_hit = ray_triangle_intersect(ray, v0, v1, v2);
        if (tri_hit.hit != 0u && tri_hit.t < result.t) {
            result = tri_hit;
        }
    }

    return result;
}

// Main hybrid intersection function
fn intersect_hybrid(ray: Ray) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    // Test mesh geometry if enabled
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    // Terrain heightfield (hybrid mix, or selectable as the primary
    // intersectable via mode 3); defined in hybrid_terrain_traversal.wgsl.
    if ((hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 3u)
        && terrain_enabled()) {
        var tray = ray;
        tray.tmax = best_hit.t;
        let terrain_hit = terrain_intersect(tray);
        if (terrain_hit.hit != 0u && terrain_hit.t < best_hit.t) {
            best_hit = terrain_hit;
        }
    }

    return best_hit;
}

// Performance-optimized early termination
fn intersect_hybrid_optimized(
    ray: Ray,
    early_exit_distance: f32,
    apply_terrain_curvature: bool,
) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < early_exit_distance) {
            return mesh_hit;
        }
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    // Terrain heightfield shadow/any-hit path shares the min-max descent.
    if ((hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 3u)
        && terrain_enabled()) {
        var tray = ray;
        tray.tmax = best_hit.t;
        let terrain_hit = terrain_trace(tray, true, apply_terrain_curvature);
        if (terrain_hit.hit != 0u && terrain_hit.t < best_hit.t) {
            best_hit = terrain_hit;
        }
    }

    return best_hit;
}

// Utility function to get surface properties at hit point
fn get_surface_properties(hit: HybridHitResult) -> vec3f {
    // Terrain hits use the uniform terrain albedo; everything else keeps the
    // legacy constant.
    if (hit.hit_type == 3u) {
        return terrain.albedo_pad.rgb;
    }
    return vec3f(0.7, 0.7, 0.8);
}

// Shadow ray testing for both SDF and mesh geometry
fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool {
    let hit = intersect_hybrid_optimized(ray, 0.01, true);
    return hit.hit != 0u && hit.t < max_distance;
}

// IBL rays are arbitrary hemisphere samples, so the Sun's azimuth-specific
// effective curvature radius is not physically applicable to their terrain
// occlusion test.
fn intersect_ibl_occlusion_ray(ray: Ray, max_distance: f32) -> bool {
    let hit = intersect_hybrid_optimized(ray, 0.01, false);
    return hit.hit != 0u && hit.t < max_distance;
}

// Test occlusion for soft shadows (SDF can provide smoother shadows)
fn soft_shadow_factor(ray: Ray, max_distance: f32, softness: f32) -> f32 {
    return select(1.0, 0.0, intersect_shadow_ray(ray, max_distance));
}
