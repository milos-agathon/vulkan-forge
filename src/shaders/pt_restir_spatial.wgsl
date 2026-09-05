// src/shaders/pt_restir_spatial.wgsl
// ReSTIR DI: Spatial reuse (MVP stub)
// For each pixel, compare its reservoir to 4-neighborhood (up, down, left, right)
// and keep the reservoir with the highest weight. Writes a diagnostic flag if
// a neighbor was adopted (bit 0 = adopted neighbor, 0 = kept self).

struct Uniforms {
    width: u32,
    height: u32,
    frame_index: u32,
    spp: u32,
    cam_origin: vec3<f32>,
    cam_fov_y: f32,
    cam_right: vec3<f32>,
    cam_aspect: f32,
    cam_up: vec3<f32>,
    cam_exposure: f32,
    cam_forward: vec3<f32>,
    seed_hi: u32,
    seed_lo: u32,
    camera_model: u32,
    full_width: u32,
    full_height: u32,
    pixel_offset_x: u32,
    pixel_offset_y: u32,
    ortho_half_height: f32,
    camera_flags: u32,
    sensor_rect: vec4<f32>,
}

// Scene lights (Group 1)
struct AreaLight {
    position: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    importance: f32,
}

struct DirectionalLight {
    direction: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    importance: f32,
}

@group(1) @binding(4) var<storage, read> area_lights: array<AreaLight>;
@group(1) @binding(5) var<storage, read> directional_lights: array<DirectionalLight>;

// Helper to process one candidate reservoir with reservoir update
fn consider_candidate(
    r: Reservoir,
    pix_idx: u32,
    source_is_self: bool,
    Wsum_ref: ptr<function, f32>,
    chosen_sample_ref: ptr<function, LightSample>,
    chosen_pdf_ref: ptr<function, f32>,
    reused_ref: ptr<function, bool>,
    seed_ref: ptr<function, u32>,
    sum_imp_area: f32,
    area_count: u32,
    sum_imp_dir: f32,
    dir_count: u32,
) {
    if (r.m == 0u) { return; }
    // Re-evaluate target pdf at current pixel using G-buffer and scene lights
    let nr = gbuffer_nr[pix_idx];
    let N = normalize(nr.xyz);
    let P = gbuffer_pos[pix_idx].xyz;
    var p_curr: f32 = 0.0;
    if (r.sample.light_type == 1u) {
        // Directional (delta): selection probability only
        if (dir_count == 0u) { return; }
        let idx = min(r.sample.light_index, dir_count - 1u);
        let imp = max(directional_lights[idx].importance, 0.0);
        let p_sel = select(1.0 / f32(dir_count), imp / max(sum_imp_dir, 1e-8), sum_imp_dir > 0.0);
        // Require surface-facing to avoid zero-contribution picks
        let wi = normalize(r.sample.direction);
        let cosTheta = max(dot(N, wi), 0.0);
        if (cosTheta <= 0.0) { return; }
        // Terrain stores luminance(albedo * configured light color) in
        // gbuffer_nr.w and tags gbuffer_pos.w negative. This re-evaluates
        // neighbor reservoirs with the receiving pixel's material while
        // preserving the shared wavefront path's historical selection pdf.
        p_curr = select(
            p_sel,
            p_sel * nr.w * cosTheta,
            gbuffer_pos[pix_idx].w < 0.0,
        );
    } else if (r.sample.light_type == 2u) {
        // Area disc: selection probability times area->solid-angle
        if (area_count == 0u) { return; }
        let idx = min(r.sample.light_index, area_count - 1u);
        let L = r.sample.position;
        let dirv = L - P;
        let d = length(dirv);
        if (d <= 1e-6) { return; }
        let wi = dirv / d;
        let cosTheta = max(dot(N, wi), 0.0);
        if (cosTheta <= 0.0) { return; }
        let nL = normalize(area_lights[idx].normal);
        let cos_on_light = max(dot(nL, -wi), 0.0);
        if (cos_on_light <= 0.0) { return; }
        let R = max(area_lights[idx].radius, 1e-6);
        let area = PI * R * R;
        let p_area = 1.0 / area;
        let p_solid = p_area * (d * d) / max(cos_on_light, 1e-6);
        let imp = max(area_lights[idx].importance, 0.0);
        let p_sel = select(1.0 / f32(area_count), imp / max(sum_imp_area, 1e-8), sum_imp_area > 0.0);
        p_curr = p_sel * p_solid;
    } else {
        return;
    }
    if (p_curr <= 0.0 || r.target_pdf <= 0.0) { return; }
    // RIS-like weight adjustment using pdf ratio
    let w = r.w_sum * (p_curr / max(r.target_pdf, 1e-6));
    if (w <= 0.0) { return; }
    *Wsum_ref = *Wsum_ref + w;
    let u = xorshift32(seed_ref);
    if (u < w / (*Wsum_ref)) {
        *chosen_sample_ref = r.sample;
        *chosen_pdf_ref = p_curr;
        *reused_ref = !source_is_self;
    }
}

struct LightSample {
    position: vec3<f32>,
    light_index: u32,
    direction: vec3<f32>,
    intensity: f32,
    light_type: u32,
    params: vec3<f32>,
}

struct Reservoir {
    sample: LightSample,
    w_sum: f32,
    m: u32,
    weight: f32,
    target_pdf: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Group 2: spatial reuse bind group
// 0: in reservoirs (read-only) - typically temporal result
// 1: out reservoirs (read_write)
@group(2) @binding(0) var<storage, read> in_reservoirs: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> out_reservoirs: array<Reservoir>;
// Scene G-buffer (from scene group)
@group(1) @binding(10) var<storage, read> gbuffer_nr: array<vec4<f32>>;   // normal.xyz, roughness
@group(1) @binding(11) var<storage, read> gbuffer_pos: array<vec4<f32>>;  // world pos.xyz, 1

fn px(ix: u32, iy: u32, W: u32, H: u32) -> u32 {
    let cx = clamp(ix, 0u, W - 1u);
    let cy = clamp(iy, 0u, H - 1u);
    return cy * W + cx;
}

fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

const PI: f32 = 3.141592653589793;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let W = uniforms.width;
    let H = uniforms.height;
    let pixel_count = W * H;
    if (idx >= pixel_count) { return; }

    let x = idx % W;
    let y = idx / W;

    // K-neighbor spatial resampling (basic reservoir update)
    let K: u32 = 8u;
    let R: u32 = 3u;

    let global_idx = (y + uniforms.pixel_offset_y) * uniforms.full_width
        + x + uniforms.pixel_offset_x;
    var seed = (uniforms.seed_hi ^ uniforms.frame_index)
        + global_idx * 1664525u + 1013904223u;

    let r_self = in_reservoirs[idx];
    var out_r: Reservoir;
    var chosen_sample: LightSample = r_self.sample;
    var chosen_pdf: f32 = r_self.target_pdf;
    var reused = false;

    var Wsum: f32 = 0.0;
    var m_total: u32 = 0u;

    // Precompute light selection sums
    let AREA_COUNT = arrayLength(&area_lights);
    let DIR_COUNT = arrayLength(&directional_lights);
    var sum_imp_area: f32 = 0.0;
    for (var i: u32 = 0u; i < AREA_COUNT; i = i + 1u) { sum_imp_area = sum_imp_area + max(area_lights[i].importance, 0.0); }
    var sum_imp_dir: f32 = 0.0;
    for (var i: u32 = 0u; i < DIR_COUNT; i = i + 1u) { sum_imp_dir = sum_imp_dir + max(directional_lights[i].importance, 0.0); }

    // Start with self candidate
    consider_candidate(r_self, idx, true, &Wsum, &chosen_sample, &chosen_pdf, &reused, &seed, sum_imp_area, AREA_COUNT, sum_imp_dir, DIR_COUNT);
    m_total = m_total + r_self.m;

    // Full-sensor poster tiles use self-only spatial resampling so an edge
    // pixel has identical history in a tile and a monolithic frame. Temporal
    // reuse remains active and the finalized reservoir still shades beauty.
    if (uniforms.camera_flags == 0u) {
        for (var i: u32 = 0u; i < K; i = i + 1u) {
            // Uniform in [-R, R]
            let rx = i32(floor(xorshift32(&seed) * f32(2u * R + 1u))) - i32(R);
            let ry = i32(floor(xorshift32(&seed) * f32(2u * R + 1u))) - i32(R);
            if (rx == 0 && ry == 0) { continue; }
            let nx = u32(clamp(i32(x) + rx, 0, i32(W) - 1));
            let ny = u32(clamp(i32(y) + ry, 0, i32(H) - 1));
            let ni = ny * W + nx;
            let rn = in_reservoirs[ni];
            consider_candidate(rn, idx, false, &Wsum, &chosen_sample, &chosen_pdf, &reused, &seed, sum_imp_area, AREA_COUNT, sum_imp_dir, DIR_COUNT);
            m_total = m_total + rn.m;
        }
    }

    // Finalize output reservoir
    out_r.sample = chosen_sample;
    out_r.target_pdf = chosen_pdf;
    out_r.w_sum = Wsum;
    out_r.m = m_total;
    if (out_r.w_sum > 0.0 && out_r.target_pdf > 0.0) {
        out_r.weight = out_r.w_sum / (f32(out_r.m) * out_r.target_pdf);
    } else {
        out_r.weight = 0.0;
    }

    out_reservoirs[idx] = out_r;

    // Diagnostics disabled in spatial pass to avoid binding conflicts
}
