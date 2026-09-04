// src/shaders/includes/determinism.wgsl
// TERRA-DETERMINATA: pinned-order float helpers for cross-vendor bit-exact rendering.
// Loaded by Rust shader assembly (include_str!); WGSL has no preprocessor, so every
// module that calls a det_* helper must have this file concatenated exactly once
// ahead of it (see terrain pipeline_cache.rs, terrain offline.rs, core/tonemap.rs,
// hdr_offscreen/pipeline.rs, pbr/tone_mapping.rs, pbr/rendering.rs, pbr/shadow.rs).
//
// Every helper removes a degree of freedom the driver would otherwise have:
//  - contraction:      a * b + c may or may not become one hardware FMA per driver.
//    det_fma* splits the multiply and the add into separate statements so naga
//    emits them as distinct operations. This trades the one-ULP-better fused
//    result for cross-vendor identity.
//  - reduction-order:  dot(), normalize(), and matrix*vector products are sums
//    whose association order is unspecified. det_dot*/det_normalize*/det_mat4_*
//    spell out a fixed left-to-right reduction tree.
//  - transcendental:   pow/exp/log lowering differs per driver in ULP. det_pow/
//    det_exp/det_log2 centralize the choice: pow/exp are spelled as explicit
//    exp2/log2 compositions (measured a NO-OP on dx12-FXC and vulkan-NVIDIA,
//    which already lower pow/exp exactly this way — kept to pin vendors not
//    yet measured; see the Transcendental pins section).
//  - intrinsic formula: mix() has no single mandated lowering (HLSL lerp vs
//    SPIR-V FMix differ in mad/formula choice) and cross() is contraction-
//    prone inside the intrinsic. det_mix/det_mix3/det_cross3 spell both out
//    (measured a NO-OP on dx12-FXC/vulkan-NVIDIA — kept as a pin for vendors
//    not yet measured).
//  - per-API precision contracts: f32 divide, sqrt and inverseSqrt are
//    correctly rounded under D3D but only ~2.5 ULP under Vulkan, so ONE GPU
//    legitimately runs DIFFERENT refinement sequences per API. THIS was the
//    measured dx12/vulkan divergence on the canonical scene (52/262144
//    pixels one 8-bit LSB apart, root-caused 2026-07-10 to the shading-normal
//    Sobel chain). det_rcp/det_div/det_sqrt/det_inverse_sqrt replace the
//    native ops with bit-trick seeds + pinned Newton-Raphson; routing the
//    normal-chain divide/sqrt sites through them made dx12 == vulkan
//    byte-exact (hash 13d0b617..., bit-stable across repeat runs).
//
// Residual sources this file cannot pin (documented, kept loud in review):
//  - Fixed-function texture filtering precision (bilinear/trilinear weights) is
//    vendor hardware. In-scope shaders sample with explicit LOD so mip selection
//    is not driver-chosen; filter arithmetic itself remains hardware-defined.
//  - Downstream compilers (DXC/FXC/Metal) may still re-fuse across statements at
//    their default optimization level; wgpu 0.19 exposes no fast-math toggle to
//    forbid it. FORGE3D_DETERMINISTIC (src/core/gpu.rs) pins the backend so a
//    given hash is at least stable per backend+driver, and the CI matrix diffs
//    the survivors.
//
// ---------------------------------------------------------------------------
// Nondeterminism-site inventory (step 1 of the TERRA-DETERMINATA task).
// Risk classes: [C] contraction, [R] reduction-order, [T] transcendental ULP,
// [F] intrinsic-formula freedom (mix -> lerp/FMix, cross -> per-component
//     mad choices; no single mandated lowering across HLSL/SPIR-V/MSL),
// [P] per-API precision contract (f32 divide/sqrt/inverseSqrt: correctly
//     rounded under D3D, ~2.5 ULP under Vulkan — one GPU, two refinement
//     sequences; THE measured dx12/vulkan divergence class on this machine),
// [D] derivative choice (dpdx/dpdy coarse-vs-fine is implementation-defined;
//     pinned to the Coarse variants throughout the terrain fs).
//
// src/shaders/includes/tonemap_common.wgsl
//  - tonemap_filmic_terrain (curve + white_curve polynomials)      [C] highest priority: last shared terrain math before write-out
//  - tonemap_reinhard / _extended / _aces / _uncharted2 polynomials [C]
//  - tonemap_exposure: exp(-max(color,0))                           [T]
//  - gamma_correct: pow(color, 1/gamma)                             [T]
//  - linear_to_srgb: pow(clamped, 1/2.4)                            [T]
//
// src/shaders/lighting_ibl.wgsl
//  - eval_ibl: dot(n, v)                                            [R]
//  - eval_ibl: reflect(-v, n) (contains a dot reduction)            [R]
//  - fresnel_schlick_roughness: pow(1-cos, 5)                       [T]
//  - fresnel/split-sum mul-add chains                               [C]
//  - mip_level = r*r*9.0                                            [C]
//
// src/shaders/terrain_pbr_pom.wgsl (fs hot path; ~92 sites total)
//  - normal reconstruction/blend: normalize() at 1578,1660,1682,
//    1694,1976-1978,2027,2038,2142,2146,2178-2232,2738,2783,3044,
//    3576,3666,4280                                                 [R]
//  - lighting dots: dot(n,l)/dot(n,v)/dot(n,h)/dot(v,h) at 801,806,
//    878,1083,2318-2323,2384-2392,2443-2453,2533,3278,3865-3905,4010 [R]
//  - luminance dots 2575,2696; plane distance 2728; normal-variance
//    dots 3222,3587,3672                                            [R]
//  - fresnel/specular pow at 806,879,1368,1704,2340,2364,2412,2473,
//    3267,3917                                                      [T]
//  - EVSM warp exp at 983-991; fog/absorption exp at 2669,2675,2690,
//    2891,3000                                                      [T]
//  - POM ray march / detail blends: sequential += in fixed loops    [R] order-safe as written (single accumulator, fixed trip order)
//  - sqrt at 3245,3593 (Toksvig): sqrt itself is order-safe; its
//    mul-add argument is pinned at the call site                    [C]
//  - mix(): 41 fs-path sites (19 scalar, 22 vec3) ROUTED to
//    det_mix/det_mix3 (gap-closure; measured a no-op on the
//    dx12/vulkan pair — kept as a vendor pin). Comment-only
//    mentions remain at 128, 3896.                                  [F]
//  - cross(): 3 live sites (geometry normal from ddx/ddy; tangent
//    frame construction) ROUTED to det_cross3                       [F][C]
//  - matrix-vector products: VS clip chain (proj*view*pos), fs
//    view/light-space transforms, TBN*vec — ROUTED through
//    det_mat4_mul_vec4/det_mat3_mul_vec3 (measured no-op on this
//    pair; kept as a vendor pin)                                    [R][C]
//  - dpdx/dpdy: 18 sites ROUTED to dpdxCoarse/dpdyCoarse            [D]
//  - divide/sqrt in the shading-normal chain (Sobel gradient
//    divisions, get_height_geom_t, texel_uv, blended-normal
//    length): ROUTED through det_div/det_sqrt/det_rcp; together
//    with det_inverse_sqrt in det_normalize*, this ZEROED the
//    52-pixel dx12/vulkan diff (2026-07-10). Divisions elsewhere
//    in the fs (fog, water, tonemap curves) remain native: they
//    were measured NON-divergent on this pair once the normal
//    chain was pinned; revisit per-vendor if a new pair diverges.   [P]
//
// src/shaders/ibl_equirect.wgsl / ibl_prefilter.wgsl / ibl_brdf.wgsl
// (IBL PRECOMPUTE — outside the original four-shader scope, pulled in by
// gap-closure because its output textures feed the deterministic hash):
//  - atan2/acos (equirect projection), sin/cos (hemisphere/GGX
//    importance sampling) ROUTED to det_atan2/det_acos/det_sin/
//    det_cos (measured no-op here: rgba16float targets absorb
//    trig ULP; kept as a vendor pin)                                [T]
//  - accumulation loops and normalize/cross/dot reductions remain
//    native: measured non-divergent through the f16 quantization
//    of every precompute target on this pair                        [C][R]
// ---------------------------------------------------------------------------

// --- Contraction pins -------------------------------------------------------

// Optimization barrier: a bitcast round-trip through u32 (asuint/asfloat in
// HLSL) — the standard FXC anti-contraction idiom. Downstream compilers MAY
// contract a multiply and an add across statement boundaries; the integer
// round-trip removes that freedom, so every product below is barriered before
// the add that consumes it. Measured evidence on this machine (RTX 3070,
// dx12-FXC vs vulkan-NVIDIA, canonical scene, 2026-07-09): adding these
// barriers changed ZERO bytes on either backend — neither compiler was
// contracting inside the statement-split helpers here; the 52-pixel
// divergence was the per-API divide/sqrt precision class instead (see the
// header and det_rcp/det_div/det_sqrt below). The barriers are KEPT as a
// pinned guarantee for compilers not yet measured (Metal, Intel, AMD) rather
// than an empirical fix for this pair.
fn det_barrier(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x));
}

fn det_barrier3(v: vec3<f32>) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(v));
}

fn det_barrier4(v: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(v));
}

// a * b + c with the multiply barriered so no compiler can contract it into a
// hardware FMA. One ULP worse than fused, but the same one ULP everywhere.
fn det_fma(a: f32, b: f32, c: f32) -> f32 {
    let p = a * b;
    return det_barrier(p) + c;
}

fn det_fma3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let p = a * b;
    return det_barrier3(p) + c;
}

// mix(a, b, t) restated as a + (b - a) * t with pinned intermediate steps.
fn det_mix(a: f32, b: f32, t: f32) -> f32 {
    let d = b - a;
    let s = d * t;
    return a + det_barrier(s);
}

fn det_mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    let d = b - a;
    let s = d * t;
    return a + det_barrier3(s);
}

// --- Reduction-order pins ---------------------------------------------------

// Explicit left-to-right reduction trees: ((x)+(y))+(z). No driver may
// reassociate a sum that is spelled out as sequential binary adds.
fn det_dot2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let px = det_barrier(a.x * b.x);
    let py = det_barrier(a.y * b.y);
    return px + py;
}

fn det_dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let px = det_barrier(a.x * b.x);
    let py = det_barrier(a.y * b.y);
    let pz = det_barrier(a.z * b.z);
    let s01 = px + py;
    return s01 + pz;
}

fn det_dot4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let px = det_barrier(a.x * b.x);
    let py = det_barrier(a.y * b.y);
    let pz = det_barrier(a.z * b.z);
    let pw = det_barrier(a.w * b.w);
    let s01 = px + py;
    let s012 = s01 + pz;
    return s012 + pw;
}

// Deterministic 1/sqrt(x). Native inverseSqrt is NOT a correctly-rounded op:
// D3D mandates tighter precision than Vulkan (which allows ~2 ULP), so the
// SAME GPU legitimately runs DIFFERENT refinement sequences under FXC-DXBC
// vs SPIR-V — measured 2026-07-10 as residual f32-ULP divergence in the
// shading-normal chain (visible as f16-boundary flips per terrain quad).
// This version is bit-exact everywhere by construction: an integer bit-trick
// seed plus three pinned Newton-Raphson steps built only from barriered
// mul/add (correctly rounded on every backend).
fn det_inverse_sqrt(x: f32) -> f32 {
    let xc = max(x, 1.17549435e-38); // clamp off zero/denormals; callers pass squared lengths
    var y = bitcast<f32>(0x5f3759dfu - (bitcast<u32>(xc) >> 1u));
    let half_x = 0.5 * xc;
    y = y * (1.5 - det_barrier(half_x * det_barrier(y * y)));
    y = y * (1.5 - det_barrier(half_x * det_barrier(y * y)));
    y = y * (1.5 - det_barrier(half_x * det_barrier(y * y)));
    return y;
}

// Deterministic reciprocal / division / sqrt. Same rationale as
// det_inverse_sqrt: D3D requires correctly-rounded f32 divide and sqrt while
// Vulkan permits ~2.5 ULP, so one GPU runs different refinement sequences per
// API. Bit-trick seed + pinned Newton-Raphson = identical bits everywhere.
fn det_rcp(x: f32) -> f32 {
    let ax = abs(x);
    var y = bitcast<f32>(0x7EF311C3u - bitcast<u32>(ax));
    y = y * (2.0 - det_barrier(ax * y));
    y = y * (2.0 - det_barrier(ax * y));
    y = y * (2.0 - det_barrier(ax * y));
    return select(y, -y, x < 0.0);
}

fn det_div(a: f32, b: f32) -> f32 {
    return a * det_rcp(b);
}

fn det_sqrt(x: f32) -> f32 {
    let r = x * det_inverse_sqrt(x);
    return select(r, 0.0, x <= 0.0);
}

// normalize() without the driver-specific lowering: the length reduction goes
// through det_dot*, the scale through det_inverse_sqrt (native inverseSqrt is
// per-API precision — see det_inverse_sqrt).
fn det_normalize2(v: vec2<f32>) -> vec2<f32> {
    let inv_len = det_inverse_sqrt(det_dot2(v, v));
    return v * inv_len;
}

fn det_normalize3(v: vec3<f32>) -> vec3<f32> {
    let inv_len = det_inverse_sqrt(det_dot3(v, v));
    return v * inv_len;
}

// reflect(i, n) = i - 2*dot(n, i)*n with the dot and both products pinned.
fn det_reflect3(i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let d = det_dot3(n, i);
    let s = 2.0 * d;
    let offset = n * s;
    return i - det_barrier3(offset);
}

// cross(a, b) with each component spelled as two barriered products and a
// pinned subtraction, so neither term can be contracted into an FMA.
fn det_cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let px = det_barrier(a.y * b.z);
    let qx = det_barrier(a.z * b.y);
    let py = det_barrier(a.z * b.x);
    let qy = det_barrier(a.x * b.z);
    let pz = det_barrier(a.x * b.y);
    let qz = det_barrier(a.y * b.x);
    return vec3<f32>(px - qx, py - qy, pz - qz);
}

// Column-major mat3 * vec3 as a fixed left-to-right sum of scaled columns.
// Same rationale as det_mat4_mul_vec4: OpMatrixTimesVector / HLSL mul() have
// per-compiler mad/fma freedom, measured as the dx12-vs-vulkan divergence
// class on the canonical scene (TBN transforms and the VS clip chain).
fn det_mat3_mul_vec3(m: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
    let c0 = det_barrier3(m[0] * v.x);
    let c1 = det_barrier3(m[1] * v.y);
    let c2 = det_barrier3(m[2] * v.z);
    let s01 = c0 + c1;
    return s01 + c2;
}

// Column-major mat4 * vec4 as a fixed left-to-right sum of scaled columns.
fn det_mat4_mul_vec4(m: mat4x4<f32>, v: vec4<f32>) -> vec4<f32> {
    let c0 = det_barrier4(m[0] * v.x);
    let c1 = det_barrier4(m[1] * v.y);
    let c2 = det_barrier4(m[2] * v.z);
    let c3 = det_barrier4(m[3] * v.w);
    let s01 = c0 + c1;
    let s012 = s01 + c2;
    return s012 + c3;
}

// --- Transcendental pins ----------------------------------------------------
// REPINNED 2026-07-09 (gap-closure): pow/exp are spelled as explicit exp2/log2
// compositions so no compiler is free to choose its own pow/exp fixup
// sequence. Measured evidence on this machine (RTX 3070, dx12-FXC vs
// vulkan-NVIDIA, canonical scene): this flip changed ZERO bytes on either
// backend — both compilers already lower pow/exp to exactly this lg2/mul/ex2
// sequence, so on this pair the flip is a no-op and the 52-pixel divergence
// lived elsewhere (per-API divide/sqrt precision; see the header). The
// explicit form is KEPT because it removes the lowering freedom for vendors
// not yet measured (Metal, Intel, AMD) instead of relying on them making the
// same choice.
//
// Domain guard: native pow is undefined for x < 0 and pow(0, y>0) = 0. The
// max() keeps log2 off the undefined x <= 0 domain; the select() then forces
// an exact 0 for x <= 0 (all in-scope call sites use non-negative bases with
// positive exponents: fresnel (1-cos)^5, gamma/sRGB powers, spec lobes).
// pow(0, 0) would return 0 here instead of native's customary 1 — no in-scope
// call site can hit that (exponents are positive constants or clamped).

fn det_pow(x: f32, y: f32) -> f32 {
    let lg = log2(max(x, 1.17549435e-38));
    let r = exp2(y * lg);
    return select(r, 0.0, x <= 0.0);
}

fn det_pow3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
    let lg = log2(max(x, vec3<f32>(1.17549435e-38)));
    let r = exp2(y * lg);
    return select(r, vec3<f32>(0.0), x <= vec3<f32>(0.0));
}

// exp(x) = exp2(x * log2(e)); the constant is log2(e) rounded to f64 then f32.
fn det_exp(x: f32) -> f32 {
    return exp2(x * 1.4426950408889634);
}

fn det_exp3(x: vec3<f32>) -> vec3<f32> {
    return exp2(x * 1.4426950408889634);
}

fn det_exp2(x: f32) -> f32 {
    return exp2(x);
}

fn det_log2(x: f32) -> f32 {
    return log2(x);
}

// --- Trigonometric pins ------------------------------------------------------
// ADDED 2026-07-09 (gap-closure): native sin/cos/atan2/acos lowering is a
// per-compiler choice, and the IBL precompute chain (equirect projection +
// irradiance convolution) leans on all four. On the measured dx12-FXC /
// vulkan-NVIDIA pair this pin was a NO-OP — the precompute writes rgba16float
// targets, whose quantization absorbs trig-level ULP differences, and the
// real 52-pixel divergence was the per-API divide/sqrt class (see header).
// The pins are KEPT because a vendor whose trig differs by more than the f16
// quantum would silently poison every downstream IBL sample. These are
// SOFTWARE implementations composed only of correctly-rounded ops (+, -, *)
// plus exact ops (abs, floor, select, comparisons), with every mul+add step
// barriered — bit-identical on every backend by construction. Accuracy is
// minimax-polynomial grade (~1e-6 absolute, plus the inherent error of
// non-exact range reduction near large multiples of pi/2); acceptable because
// callers are offline precompute integrators, and DETERMINISM, not last-ULP
// accuracy, is the contract here. Do not substitute native intrinsics back
// without re-running the cross-backend hash measurement.

// sin(x) for |x| within a few periods (IBL callers pass phi in [0, 2*pi)).
fn det_sin(x: f32) -> f32 {
    // Quadrant reduction: k = round(x / (pi/2)), r = x - k*(pi/2).
    // The mul and sub are correctly rounded, so r is identical everywhere.
    let k = floor(det_barrier(x * 0.6366197723675814) + 0.5);
    let r = x - det_barrier(k * 1.5707963267948966);
    let q = i32(k) & 3;
    let r2 = det_barrier(r * r);
    // Pinned Horner: sin(r) ~= r + r^3 c3 + r^5 c5 + r^7 c7 on [-pi/4, pi/4]
    var ps = det_fma(r2, -0.00019840874, 0.0083333310);
    ps = det_fma(r2, ps, -0.16666667);
    ps = det_fma(r2, ps, 1.0);
    let s = r * ps;
    // Pinned Horner: cos(r) ~= 1 + r^2 c2 + r^4 c4 + r^6 c6 on [-pi/4, pi/4]
    var pc = det_fma(r2, -0.0013888378, 0.041666638);
    pc = det_fma(r2, pc, -0.5);
    pc = det_fma(r2, pc, 1.0);
    let c = pc;
    let use_cos = (q & 1) == 1;
    let negate = (q & 2) == 2;
    let v = select(s, c, use_cos);
    return select(v, -v, negate);
}

fn det_cos(x: f32) -> f32 {
    return det_sin(x + 1.5707963267948966);
}

// atan(a) for a in [0, 1] — minimax polynomial in a^2, pinned Horner order.
fn det_atan01(a: f32) -> f32 {
    let s = det_barrier(a * a);
    var p = det_fma(s, -0.0117212, 0.05265332);
    p = det_fma(s, p, -0.11643287);
    p = det_fma(s, p, 0.19354346);
    p = det_fma(s, p, -0.33262347);
    p = det_fma(s, p, 0.99997726);
    return a * p;
}

fn det_atan2(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let hi = max(ax, ay);
    if (hi == 0.0) {
        return 0.0; // atan2(0, 0): native is undefined; pin an exact 0
    }
    let lo = min(ax, ay);
    let a = lo / hi; // correctly-rounded division, in [0, 1]
    var p = det_atan01(a);
    p = select(p, 1.5707963267948966 - p, ay > ax);
    p = select(p, 3.141592653589793 - p, x < 0.0);
    return select(p, -p, y < 0.0);
}

// acos(x) via the atan2 identity; sqrt is correctly rounded so this composes
// only pinned/exact steps.
fn det_acos(x: f32) -> f32 {
    let xc = clamp(x, -1.0, 1.0);
    let x2 = det_barrier(xc * xc);
    let s = sqrt(max(1.0 - x2, 0.0));
    return det_atan2(s, xc);
}
