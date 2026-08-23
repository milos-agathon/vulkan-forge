// src/shaders/hybrid_terrain_traversal.wgsl
// PROMETHEUS: heightfield-native ray traversal for the hybrid path tracer.
// Implements a min-max quadtree DDA over the RG32Float pyramid built by
// terrain_heightfield.rs: descend from the coarsest mip, skip any node whose
// ray segment lies entirely above max_height or below min_height, refine into
// children only where the ray brackets the height band, and solve the exact
// ray/bilinear-patch intersection at leaf cells (the vertical deviation along
// the ray is exactly quadratic in t). Primary and sun-shadow rays reuse the
// identical descent. Also defines `main_terrain`, the accumulation-aware
// kernel entry used by HybridPathTracer::render_terrain_reference, and
// `main_terrain_gbuffer`, which feeds the pt_restir_spatial reuse pass.
// Concatenated into the hybrid kernel module by hybrid_compute/setup.rs; uses
// structs/bindings declared in hybrid_traversal.wgsl and hybrid_kernel.wgsl.
// RELEVANT FILES: src/path_tracing/hybrid_compute/terrain_heightfield.rs,
//                 src/path_tracing/hybrid_compute/render_terrain.rs

// Layout documented in terrain_heightfield.rs (six vec4 rows, 96 bytes).
struct TerrainPtUniforms {
    origin_spacing: vec4<f32>, // origin_x, origin_z, spacing_x, spacing_z
    h_params: vec4<f32>,       // h_min, h_max, exaggeration, env_intensity
    albedo_pad: vec4<f32>,     // terrain albedo rgb, unused
    dims: vec4<u32>,           // width_texels, height_texels, cell_w, cell_h
    mips: vec4<u32>,           // mip_count, flags(bit0 enabled), env_w, env_h
    extra: vec4<u32>,          // spp, welford_window, unused, unused
}

struct EarthCurvatureUniforms {
    inv_two_r_prime: f32,
    _pad0: f32,
    ray_origin_geodetic: vec2<f32>,
    enabled: u32,
    _pad1: u32,
}

// Canonical ReSTIR DI structs — byte-compatible with the Rust `Reservoir` /
// `LightSample` in src/path_tracing/restir/types.rs and with the standalone
// pt_restir_temporal.wgsl / pt_restir_spatial.wgsl reuse passes that this
// path dispatches between accumulation frames (storage stride 80 bytes).
struct RestirLightSample {
    position: vec3<f32>,
    light_index: u32,
    direction: vec3<f32>,
    intensity: f32,
    light_type: u32,   // 0=point, 1=directional, 2=area
    params: vec3<f32>,
}

struct RestirReservoir {
    sample: RestirLightSample,
    w_sum: f32,
    m: u32,
    weight: f32,       // W = w_sum / (M * target_pdf)
    target_pdf: f32,
}

@group(2) @binding(1) var terrain_height_tex: texture_2d<f32>;
@group(2) @binding(2) var terrain_minmax_tex: texture_2d<f32>;
@group(2) @binding(3) var<uniform> terrain: TerrainPtUniforms;
@group(2) @binding(4) var<storage, read_write> terrain_welford: array<vec2<f32>>;
// Fresh per-frame light candidates (input to the temporal reuse pass).
@group(2) @binding(5) var<storage, read_write> terrain_reservoirs_curr: array<RestirReservoir>;
@group(2) @binding(6) var terrain_env_tex: texture_2d<f32>;
// Reservoirs merged by last frame's temporal+spatial passes; shading reads
// them and M-clamps in place (standard ReSTIR history clamp).
@group(2) @binding(7) var<storage, read_write> terrain_reservoirs_prev: array<RestirReservoir>;
// ReSTIR G-buffer consumed by pt_restir_spatial.wgsl for target-pdf
// re-evaluation at the receiving pixel. Written once (static scene) by the
// main_terrain_gbuffer entry, which has its own pipeline layout.
@group(2) @binding(8) var<storage, read_write> terrain_gbuffer_nr: array<vec4<f32>>;
@group(2) @binding(9) var<storage, read_write> terrain_gbuffer_pos: array<vec4<f32>>;
@group(2) @binding(10) var<uniform> earth_curvature: EarthCurvatureUniforms;

const TERRAIN_STACK_SIZE: u32 = 64u;
const TERRAIN_PI: f32 = 3.14159265358979323846;
// ReSTIR history cap: prev reservoirs are rescaled to at most this M before
// each temporal merge so w_sum/M cannot blow up across hundreds of frames.
const TERRAIN_RESTIR_M_CAP: u32 = 512u;

fn terrain_reservoir_weight(w_sum: f32, m: u32, target_pdf: f32) -> f32 {
    return w_sum / (f32(m) * target_pdf);
}

fn terrain_enabled() -> bool {
    return (terrain.mips.y & 1u) != 0u;
}

// Safe reciprocal that avoids inf propagation for axis-parallel rays.
fn terrain_safe_inv(d: f32) -> f32 {
    let ad = max(abs(d), 1e-12);
    return select(1.0 / ad, -1.0 / ad, d < 0.0);
}

// Camera-relative formulation: t * direction.xz is the accumulated horizontal
// ray displacement, so d² never subtracts large world coordinates in f32.
fn terrain_curved_height(ray: Ray, t: f32, apply_curvature: bool) -> f32 {
    let horizontal_d2 = t * t * dot(ray.direction.xz, ray.direction.xz);
    let correction = select(
        0.0,
        horizontal_d2 * earth_curvature.inv_two_r_prime,
        apply_curvature && earth_curvature.enabled != 0u,
    );
    return ray.origin.y + t * ray.direction.y + correction;
}

// The corrected ray height is a convex quadratic. Endpoints bound its maximum;
// including the in-span vertex gives the exact minimum, hence node rejection
// is conservative without widening or duplicating the min-max pyramid.
fn terrain_curved_height_range(
    ray: Ray,
    t0: f32,
    t1: f32,
    apply_curvature: bool,
) -> vec2<f32> {
    let y0 = terrain_curved_height(ray, t0, apply_curvature);
    let y1 = terrain_curved_height(ray, t1, apply_curvature);
    var minimum = min(y0, y1);
    if (apply_curvature && earth_curvature.enabled != 0u) {
        let a = dot(ray.direction.xz, ray.direction.xz) * earth_curvature.inv_two_r_prime;
        if (a > 0.0) {
            let vertex = -ray.direction.y / (2.0 * a);
            if (vertex >= t0 && vertex <= t1) {
                minimum = min(minimum, terrain_curved_height(ray, vertex, true));
            }
        }
    }
    return vec2<f32>(minimum, max(y0, y1));
}

// Ray parameter span over the world-space xz rectangle of a node.
// Returns (t_enter, t_exit); empty if t_enter > t_exit.
fn terrain_slab_xz(ray: Ray, x0: f32, x1: f32, z0: f32, z1: f32) -> vec2<f32> {
    let inv_x = terrain_safe_inv(ray.direction.x);
    let inv_z = terrain_safe_inv(ray.direction.z);
    var tx0 = (x0 - ray.origin.x) * inv_x;
    var tx1 = (x1 - ray.origin.x) * inv_x;
    if (tx0 > tx1) { let tmp = tx0; tx0 = tx1; tx1 = tmp; }
    var tz0 = (z0 - ray.origin.z) * inv_z;
    var tz1 = (z1 - ray.origin.z) * inv_z;
    if (tz0 > tz1) { let tmp = tz0; tz0 = tz1; tz1 = tmp; }
    return vec2<f32>(max(tx0, tz0), min(tx1, tz1));
}

// Node (level, x, y) -> packed stack word. Supports DEMs up to 8192 cells.
fn terrain_pack_node(level: u32, x: u32, y: u32) -> u32 {
    return (level << 26u) | (y << 13u) | x;
}

// Exaggerated corner heights of DEM cell (cx, cz): h00, h10, h01, h11.
fn terrain_cell_heights(cx: u32, cz: u32) -> vec4<f32> {
    let ex = terrain.h_params.z;
    let h00 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx), i32(cz)), 0).r;
    let h10 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx + 1u), i32(cz)), 0).r;
    let h01 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx), i32(cz + 1u)), 0).r;
    let h11 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx + 1u), i32(cz + 1u)), 0).r;
    return vec4<f32>(h00, h10, h01, h11) * ex;
}

struct TerrainLeafHit {
    t: f32,
    hit: bool,
}

// Exact ray vs bilinear-patch test inside cell (cx, cz) over ray span
// [t0, t1]. The vertical deviation d(t) = ray_y(t) - H(t) is exactly
// quadratic in t (H is bilinear, the ray footprint is linear), so fit the
// quadratic through d(t0), d(mid), d(t1) and take its smallest root in range.
fn terrain_leaf_intersect(
    ray: Ray,
    cx: u32,
    cz: u32,
    t0: f32,
    t1: f32,
    apply_curvature: bool,
    any_hit: bool,
) -> TerrainLeafHit {
    var out: TerrainLeafHit;
    out.hit = false;
    out.t = 1e30;
    let h = terrain_cell_heights(cx, cz);
    let sx = terrain.origin_spacing.z;
    let sz = terrain.origin_spacing.w;
    let ox = terrain.origin_spacing.x;
    let oz = terrain.origin_spacing.y;

    let tm = 0.5 * (t0 + t1);
    var d3: vec3<f32>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let t = select(select(t1, tm, i == 1u), t0, i == 0u);
        let px = ray.origin.x + t * ray.direction.x;
        let pz = ray.origin.z + t * ray.direction.z;
        let u = clamp((px - ox) / sx - f32(cx), 0.0, 1.0);
        let v = clamp((pz - oz) / sz - f32(cz), 0.0, 1.0);
        let hh = mix(mix(h.x, h.y, u), mix(h.z, h.w, u), v);
        d3[i] = terrain_curved_height(ray, t, apply_curvature) - hh;
    }

    // d(s) = a s^2 + b s + c on s in [0,1] with s = (t - t0)/(t1 - t0).
    let c = d3.x;
    let a = 2.0 * d3.z + 2.0 * d3.x - 4.0 * d3.y;
    let b = d3.z - d3.x - a;

    // A rounded shared-cell boundary can place this leaf's entry infinitesimally
    // below the continuous surface. That is already an any-hit intersection;
    // requiring another crossing inside the leaf would create a numeric crack.
    var s_hit = 1e30;
    if (any_hit && c <= 0.0) {
        s_hit = 0.0;
    } else if (abs(a) < 1e-12) {
        if (abs(b) > 1e-12) {
            let s = -c / b;
            if (s >= 0.0 && s <= 1.0) { s_hit = s; }
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            // Numerically stable quadratic roots (Citardauq form for the
            // second root avoids cancellation).
            let sq = sqrt(disc);
            let q = -0.5 * (b + select(-sq, sq, b >= 0.0));
            var r0 = q / a;
            var r1 = select(c / q, 1e30, abs(q) < 1e-30);
            if (r0 > r1) { let tmp = r0; r0 = r1; r1 = tmp; }
            if (r0 >= 0.0 && r0 <= 1.0) { s_hit = r0; }
            else if (r1 >= 0.0 && r1 <= 1.0) { s_hit = r1; }
        }
    }
    if (s_hit <= 1.0) {
        let t = t0 + s_hit * (t1 - t0);
        if (t > ray.tmin && t < ray.tmax) {
            out.hit = true;
            out.t = t;
        }
    }
    return out;
}

// Geometric normal from the analytic bilinear gradient at world point p in
// cell (cx, cz) — the same surface the leaf test intersected.
fn terrain_normal_at(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {
    let h = terrain_cell_heights(cx, cz);
    let sx = terrain.origin_spacing.z;
    let sz = terrain.origin_spacing.w;
    let u = clamp((p.x - terrain.origin_spacing.x) / sx - f32(cx), 0.0, 1.0);
    let v = clamp((p.z - terrain.origin_spacing.y) / sz - f32(cz), 0.0, 1.0);
    let dh_du = mix(h.y - h.x, h.w - h.z, v);
    let dh_dv = mix(h.z - h.x, h.w - h.y, u);
    return normalize(vec3<f32>(-dh_du / sx, 1.0, -dh_dv / sz));
}

// Min-max quadtree DDA. `any_hit` controls early exit and conservatively closes
// rounded shared-leaf entry cracks. Curvature is an explicit sun-ray policy
// because arbitrary IBL directions must not reuse the azimuth-specific
// effective radius uploaded for the Sun.
fn terrain_trace(ray: Ray, any_hit: bool, apply_curvature: bool) -> HybridHitResult {
    var res: HybridHitResult;
    res.hit = 0u;
    res.t = ray.tmax;
    res.hit_type = 3u; // terrain
    if (!terrain_enabled()) { return res; }

    let cell_w = terrain.dims.z;
    let cell_h = terrain.dims.w;
    let ox = terrain.origin_spacing.x;
    let oz = terrain.origin_spacing.y;
    let sx = terrain.origin_spacing.z;
    let sz = terrain.origin_spacing.w;

    var stack: array<u32, TERRAIN_STACK_SIZE>;
    var sp = 0u;
    stack[sp] = terrain_pack_node(terrain.mips.x - 1u, 0u, 0u);
    sp = sp + 1u;

    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = stack[sp];
        let level = node >> 26u;
        let ny = (node >> 13u) & 0x1FFFu;
        let nx = node & 0x1FFFu;

        // Cell range covered by this node (clamped at ragged edges).
        let cx0 = nx << level;
        let cz0 = ny << level;
        if (cx0 >= cell_w || cz0 >= cell_h) { continue; }
        let cx1 = min((nx + 1u) << level, cell_w);
        let cz1 = min((ny + 1u) << level, cell_h);

        let span = terrain_slab_xz(
            ray,
            ox + f32(cx0) * sx,
            ox + f32(cx1) * sx,
            oz + f32(cz0) * sz,
            oz + f32(cz1) * sz,
        );
        let t_lo = max(span.x, ray.tmin);
        let t_hi = min(span.y, min(ray.tmax, res.t));
        if (t_lo > t_hi) { continue; }

        // Height band test: skip when the ray segment stays entirely above
        // max or below min over this node's footprint.
        let mm = textureLoad(terrain_minmax_tex, vec2<i32>(i32(nx), i32(ny)), i32(level)).rg
            * terrain.h_params.z;
        let ray_height = terrain_curved_height_range(ray, t_lo, t_hi, apply_curvature);
        if (ray_height.x > mm.y || ray_height.y < mm.x) { continue; }

        if (level == 0u) {
            let leaf = terrain_leaf_intersect(ray, cx0, cz0, t_lo, t_hi, apply_curvature, any_hit);
            if (leaf.hit && leaf.t < res.t) {
                res.hit = 1u;
                res.t = leaf.t;
                res.point = ray.origin + ray.direction * leaf.t;
                res.normal = terrain_normal_at(res.point, cx0, cz0);
                res.material_id = 0u;
                res.hit_type = 3u;
                if (any_hit) { return res; }
            }
            continue;
        }

        // Push the (up to) four children ordered far-to-near by t_enter so
        // the nearest is popped first; empty children are skipped.
        let child_level = level - 1u;
        var child_t: array<f32, 4u>;
        var child_id: array<u32, 4u>;
        var child_count = 0u;
        for (var cy = 0u; cy < 2u; cy = cy + 1u) {
            for (var cxi = 0u; cxi < 2u; cxi = cxi + 1u) {
                let ccx = nx * 2u + cxi;
                let ccy = ny * 2u + cy;
                let gx0 = ccx << child_level;
                let gz0 = ccy << child_level;
                if (gx0 >= cell_w || gz0 >= cell_h) { continue; }
                let gx1 = min((ccx + 1u) << child_level, cell_w);
                let gz1 = min((ccy + 1u) << child_level, cell_h);
                let cs = terrain_slab_xz(
                    ray,
                    ox + f32(gx0) * sx,
                    ox + f32(gx1) * sx,
                    oz + f32(gz0) * sz,
                    oz + f32(gz1) * sz,
                );
                let ct_lo = max(cs.x, t_lo);
                let ct_hi = min(cs.y, t_hi);
                if (ct_lo > ct_hi) { continue; }
                child_t[child_count] = ct_lo;
                child_id[child_count] = terrain_pack_node(child_level, ccx, ccy);
                child_count = child_count + 1u;
            }
        }
        // Insertion sort (descending t_enter) then push: nearest ends on top.
        for (var i = 1u; i < child_count; i = i + 1u) {
            let kt = child_t[i];
            let kid = child_id[i];
            var j = i;
            loop {
                if (j == 0u || child_t[j - 1u] >= kt) { break; }
                child_t[j] = child_t[j - 1u];
                child_id[j] = child_id[j - 1u];
                j = j - 1u;
            }
            child_t[j] = kt;
            child_id[j] = kid;
        }
        for (var i = 0u; i < child_count; i = i + 1u) {
            if (sp < TERRAIN_STACK_SIZE) {
                stack[sp] = child_id[i];
                sp = sp + 1u;
            }
        }
    }
    return res;
}

fn terrain_intersect(ray: Ray) -> HybridHitResult {
    return terrain_trace(ray, false, false);
}

fn terrain_occluded(ray: Ray, max_distance: f32) -> bool {
    var r = ray;
    r.tmax = min(ray.tmax, max_distance);
    let hit = terrain_trace(r, true, true);
    return hit.hit != 0u;
}

// ---------------------------------------------------------------------------
// Environment (IBL) term
// ---------------------------------------------------------------------------
// Equirect env map lookup by direction (nearest texel — deterministic). When
// no map is bound (env_w == 0) the fallback is a constant white environment
// scaled by env_intensity: still routed through this same function so there
// is no silent divergence between the two configurations.
fn terrain_env_radiance(dir: vec3<f32>) -> vec3<f32> {
    let intensity = terrain.h_params.w;
    let ew = terrain.mips.z;
    let eh = terrain.mips.w;
    if (ew == 0u || eh == 0u) {
        return vec3<f32>(intensity);
    }
    let d = normalize(dir);
    let uu = (atan2(d.z, d.x) / (2.0 * TERRAIN_PI)) + 0.5;
    let vv = acos(clamp(d.y, -1.0, 1.0)) / TERRAIN_PI;
    let px = min(u32(uu * f32(ew)), ew - 1u);
    let py = min(u32(vv * f32(eh)), eh - 1u);
    return textureLoad(terrain_env_tex, vec2<i32>(i32(px), i32(py)), 0).rgb * intensity;
}

// Zero-mean tent sample in [-1, 1] (inverse-CDF; the kernel-local
// tent_filter in hybrid_kernel.wgsl returns the PDF value, not a sample).
fn terrain_tent_offset(u: f32) -> f32 {
    if (u < 0.5) {
        return sqrt(2.0 * u) - 1.0;
    }
    return 1.0 - sqrt(2.0 * (1.0 - u));
}

fn terrain_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Cosine-weighted hemisphere direction about n.
fn terrain_cosine_dir(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bt = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    let r = sqrt(u1);
    let phi = 2.0 * TERRAIN_PI * u2;
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    return normalize(local.x * t + local.y * bt + local.z * n);
}

// ---------------------------------------------------------------------------
// Accumulating terrain-reference kernel entry
// ---------------------------------------------------------------------------
// Per frame: `spp` jittered camera samples averaged into accum_hdr, canonical
// ReSTIR candidate generation into terrain_reservoirs_curr (merged afterwards
// by the pt_restir_temporal + pt_restir_spatial passes the driver dispatches),
// sun shading gated through the merged reservoir from the previous frame's
// reuse chain, a windowed Welford update of the running-mean luminance for
// the "variance across the last N frames" convergence gate, and the
// tonemapped running mean written to out_tex. AOVs are written from an
// UNJITTERED center ray when aov_flags is set (the driver sets it on frame 0
// only) so geometric AOVs match rasterizer pixel-center sampling.
@compute @workgroup_size(8, 8, 1)
fn main_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = uniforms.width;
    let H = uniforms.height;
    if (gid.x >= W || gid.y >= H) { return; }
    let pix = gid.y * W + gid.x;

    // --- ReSTIR history M-clamp + fetch the merged reservoir for shading ---
    var prev_r = terrain_reservoirs_prev[pix];
    if (prev_r.m > TERRAIN_RESTIR_M_CAP) {
        let scale = f32(TERRAIN_RESTIR_M_CAP) / f32(prev_r.m);
        prev_r.w_sum = prev_r.w_sum * scale;
        prev_r.m = TERRAIN_RESTIR_M_CAP;
        if (prev_r.target_pdf > 0.0) {
            prev_r.weight = terrain_reservoir_weight(prev_r.w_sum, prev_r.m, prev_r.target_pdf);
        }
        terrain_reservoirs_prev[pix] = prev_r;
    }
    let prev_valid = uniforms.frame_index > 0u && prev_r.m > 0u
        && prev_r.weight > 0.0 && prev_r.target_pdf > 0.0
        && prev_r.sample.light_type == 1u;

    var st: u32 = uniforms.seed_hi ^ (gid.x * 1664525u) ^ (gid.y * 1013904223u)
        ^ (uniforms.frame_index * 92837111u) ^ uniforms.seed_lo;
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let spp = max(terrain.extra.x, 1u);

    var frame_radiance = vec3<f32>(0.0);
    var cand: RestirReservoir; // zero-initialized: m=0 marks "no candidate"

    for (var s = 0u; s < spp; s = s + 1u) {
        let jx = terrain_tent_offset(xorshift32(&st)) * 0.5;
        let jy = terrain_tent_offset(xorshift32(&st)) * 0.5;

        // Jittered beauty ray.
        let ndc_x = ((f32(gid.x) + 0.5 + jx) / f32(W)) * 2.0 - 1.0;
        let ndc_y = (1.0 - (f32(gid.y) + 0.5 + jy) / f32(H)) * 2.0 - 1.0;
        var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
        rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
        let ray = Ray(uniforms.cam_origin, 1e-3, rd, 1e30);

        let hit = intersect_hybrid(ray);
        if (hit.hit == 0u) {
            frame_radiance = frame_radiance + terrain_env_radiance(rd);
            continue;
        }
        let n = hit.normal;
        let albedo = get_surface_properties(hit);

        // --- Sun candidate generation (streaming RIS, canonical layout).
        // One directional light => selection pdf 1, stream weight w =
        // target_pdf, so a fresh reservoir finalizes to W = 1 whenever the
        // surface faces the sun. Zero-area heightfield hits never divide by
        // a primitive-area term. ---
        let wi = normalize(lighting.light_dir);
        let ndotl = max(dot(n, wi), 0.0);
        let target_pdf = terrain_luminance(albedo * lighting.light_color * ndotl);
        if (target_pdf > 0.0) {
            cand.sample.position = hit.point;
            cand.sample.light_index = 0u;
            cand.sample.direction = wi;
            cand.sample.intensity = terrain_luminance(lighting.light_color);
            cand.sample.light_type = 1u;
            cand.w_sum = cand.w_sum + target_pdf;
            cand.m = cand.m + 1u;
            cand.target_pdf = target_pdf;
        }

        // --- Sun shading through the merged reservoir (temporal + spatial
        // reuse) from the previous frame; frame 0 falls back to the fresh
        // candidate, which is the identical delta sample with W = 1. ---
        var sun_dir = wi;
        var reuse_w = 1.0;
        if (prev_valid) {
            sun_dir = normalize(prev_r.sample.direction);
            reuse_w = clamp(prev_r.weight, 0.0, 4.0);
        }
        var sun = vec3<f32>(0.0);
        let nd = max(dot(n, sun_dir), 0.0);
        if (nd > 0.0) {
            let sray = Ray(hit.point + n * 1e-3, 1e-3, sun_dir, 1e30);
            var vis = 1.0;
            if (lighting.shadows_enabled != 0u && intersect_shadow_ray(sray, 1e30)) {
                vis = 0.0;
            }
            sun = albedo * lighting.light_color * nd * vis * reuse_w;
        }

        // --- IBL: one cosine-weighted env sample per camera sample
        // (converges by accumulation); single-sample estimator of the
        // Lambert env integral (pdf = cos/pi) is albedo * env(wi) * V. ---
        let u1 = xorshift32(&st);
        let u2 = xorshift32(&st);
        let ei = terrain_cosine_dir(n, u1, u2);
        let eray = Ray(hit.point + n * 1e-3, 1e-3, ei, 1e30);
        var env_vis = 1.0;
        if (intersect_ibl_occlusion_ray(eray, 1e30)) {
            env_vis = 0.0;
        }
        let ibl = albedo * terrain_env_radiance(ei) * env_vis;

        frame_radiance = frame_radiance + sun + ibl;
    }
    frame_radiance = frame_radiance / f32(spp);

    // Finalize + publish this frame's candidate reservoir for the reuse chain.
    if (cand.m > 0u && cand.w_sum > 0.0 && cand.target_pdf > 0.0) {
        cand.weight = terrain_reservoir_weight(cand.w_sum, cand.m, cand.target_pdf);
    }
    terrain_reservoirs_curr[pix] = cand;

    // --- Accumulate the per-frame mean radiance ---
    let prev = accum_hdr[pix];
    let acc = vec4<f32>(prev.rgb + frame_radiance, prev.a + 1.0);
    accum_hdr[pix] = acc;

    // --- Windowed Welford over the RUNNING-MEAN luminance: variance of the
    // accumulated mean across the last `welford_window` frames, the
    // "converged-when" metric from the Prometheus DoD. Resets at each window
    // boundary so the gate always measures the most recent N frames. ---
    let window = max(terrain.extra.y, 2u);
    var wf = terrain_welford[pix];
    if (uniforms.frame_index % window == 0u) { wf = vec2<f32>(0.0, 0.0); }
    let mean_lum = terrain_luminance(acc.rgb / acc.a);
    let k = f32(uniforms.frame_index % window) + 1.0;
    let delta = mean_lum - wf.x;
    let mean = wf.x + delta / k;
    let m2 = wf.y + delta * (mean_lum - mean);
    terrain_welford[pix] = vec2<f32>(mean, m2);

    // --- Resolve running mean to the output image ---
    let mean_rgb = acc.rgb / acc.a;
    let ldr = reinhard_tonemap(mean_rgb, uniforms.cam_exposure);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(ldr, 1.0));

    // --- Geometric AOVs from the unjittered center ray (frame 0 only via
    // aov_flags) so they align with rasterizer pixel-center sampling ---
    if (uniforms.aov_flags != 0u) {
        let cx = ((f32(gid.x) + 0.5) / f32(W)) * 2.0 - 1.0;
        let cy = (1.0 - (f32(gid.y) + 0.5) / f32(H)) * 2.0 - 1.0;
        var crd = normalize(vec3<f32>(cx * half_w, cy * half_h, -1.0));
        crd = normalize(crd.x * uniforms.cam_right + crd.y * uniforms.cam_up + crd.z * (-uniforms.cam_forward));
        let cray = Ray(uniforms.cam_origin, 1e-3, crd, 1e30);
        let chit = intersect_hybrid(cray);
        let is_hit = chit.hit != 0u;
        var calbedo = get_surface_properties(chit);
        if (chit.hit_type == 3u) { calbedo = terrain.albedo_pad.rgb; }
        let coord = vec2<i32>(i32(gid.x), i32(gid.y));
        if (aov_enabled(AOV_ALBEDO_BIT)) {
            textureStore(aov_albedo, coord,
                select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(calbedo, 1.0), is_hit));
        }
        if (aov_enabled(AOV_NORMAL_BIT)) {
            textureStore(aov_normal, coord,
                select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(chit.normal, 1.0), is_hit));
        }
        if (aov_enabled(AOV_DEPTH_BIT)) {
            let depth_val: f32 = select(bitcast<f32>(0x7fc00000u), chit.t, is_hit);
            textureStore(aov_depth, coord, vec4<f32>(depth_val, 0.0, 0.0, 0.0));
        }
        if (aov_enabled(AOV_VISIBILITY_BIT)) {
            textureStore(aov_visibility, coord, vec4<f32>(select(0.0, 1.0, is_hit), 0.0, 0.0, 1.0));
        }
    }
}

// ---------------------------------------------------------------------------
// ReSTIR G-buffer entry (own pipeline layout — see hybrid_compute/setup.rs)
// ---------------------------------------------------------------------------
// Writes the per-pixel surface record (world normal + roughness, world
// position) that pt_restir_spatial.wgsl re-evaluates target pdfs against.
// Camera and scene are static across the accumulation, so the driver runs
// this once before the frame loop, from the unjittered center ray.
@compute @workgroup_size(8, 8, 1)
fn main_terrain_gbuffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = uniforms.width;
    let H = uniforms.height;
    if (gid.x >= W || gid.y >= H) { return; }
    let pix = gid.y * W + gid.x;

    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let ndc_x = ((f32(gid.x) + 0.5) / f32(W)) * 2.0 - 1.0;
    let ndc_y = (1.0 - (f32(gid.y) + 0.5) / f32(H)) * 2.0 - 1.0;
    var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
    rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
    let ray = Ray(uniforms.cam_origin, 1e-3, rd, 1e30);

    let hit = intersect_hybrid(ray);
    if (hit.hit != 0u) {
        terrain_gbuffer_nr[pix] = vec4<f32>(hit.normal, 1.0);
        terrain_gbuffer_pos[pix] = vec4<f32>(hit.point, 1.0);
    } else {
        // Sky pixels: shading never consults their reservoirs; keep the
        // record finite for the spatial pass's normalize().
        terrain_gbuffer_nr[pix] = vec4<f32>(0.0, 0.0, 1.0, 1.0);
        terrain_gbuffer_pos[pix] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
