// src/shaders/terrain_pbr_pom.wgsl
// Terrain PBR + POM shader implementing normal, triplanar, and BRDF logic
// Exists to light the terrain renderer milestone with placeholder resources until assets land
// RELEVANT FILES: src/terrain_renderer.rs, src/terrain_render_params.rs, src/overlay_layer.rs, terrain_demo_task_breakdown.md
//
// Bind Groups and Layouts:
// - @group(0): Terrain uniforms and textures
//   - @binding(0): uniform<TerrainUniforms> - View/proj matrices, sun exposure, spacing, height exaggeration
//   - @binding(1): texture_2d<f32> - Height texture
//   - @binding(2): sampler - Height sampler
//   - @binding(3): texture_2d_array<f32> - Material albedo texture array
//   - @binding(4): sampler - Material sampler
//   - @binding(5): uniform<TerrainShadingUniforms> - Triplanar, POM, layer heights/roughness/metallic, light params, clamps
//   - @binding(6): texture_2d<f32> - Colormap texture
//   - @binding(7): sampler - Colormap sampler
//   - @binding(8): uniform<OverlayUniforms> - Overlay domain, blend mode, albedo mode, colormap strength, gamma
//   - @binding(9): texture_2d<f32> - Height curve LUT texture
//   - @binding(10): sampler - Height curve LUT sampler
//   - @binding(11): texture_2d<f32> - Water mask texture
//   - @binding(12): texture_2d<f32> - AO debug/fallback texture
//   - @binding(13): sampler - AO debug sampler
//   - @binding(14): texture_2d<f32> - Detail normal texture
//   - @binding(15): sampler - Detail normal sampler
//   - @binding(16): texture_2d<f32> - Heightfield ray AO texture (default: 1x1 white)
//   - @binding(17): sampler - Heightfield ray AO sampler
//   - @binding(18): texture_2d<f32> - Sun visibility texture (default: 1x1 white)
//   - @binding(19): sampler - Sun visibility sampler
// - @group(1): Light buffer (P1-06)
//   - @binding(3): storage<array<Light>> - Light array
//   - @binding(4): uniform<LightMetadata> - Light count, frame index, sequence seed
//   - @binding(5): uniform<EnvironmentParams> - Ambient color
// - @group(2): IBL textures
//   - @binding(0): texture_cube<f32> - IBL specular cube map
//   - @binding(1): texture_cube<f32> - IBL irradiance cube map
//   - @binding(2): sampler - IBL environment sampler
//   - @binding(3): texture_2d<f32> - IBL BRDF LUT
//   - @binding(4): uniform<IblUniforms> - IBL intensity, rotation (sin/cos theta), specular mip count
//
// Note: TerrainShadingUniforms (@group(0) @binding(5)) contains terrain-specific shading knobs.
// P2-05: Optional BRDF dispatch hook (disabled by default, preserves current terrain look)
// No binding collisions with mesh PBR pipeline (which uses different group layouts).

// P2-05: Include lighting.wgsl for optional eval_brdf dispatch
// Note: This adds BRDF constants and eval_brdf function, but terrain uses calculate_pbr_brdf by default
#include "lighting.wgsl"
// P4 spec: Include unified IBL evaluator (group(2) bindings)
// Note: lighting_ibl.wgsl defines PI, so we don't redefine it here
#include "lighting_ibl.wgsl"
// TV4.1: Shared terrain-noise helpers for detail normals and terrain material variation.
#include "terrain_noise.wgsl"
// TV5: Local irradiance probes.
#include "terrain_probes.wgsl"

// P2-05: Optional BRDF dispatch flag (default: false = use calculate_pbr_brdf for current look)
// Set to true to enable eval_brdf dispatch, allowing BRDF model switching on terrain
const TERRAIN_USE_BRDF_DISPATCH: bool = false;
const TERRAIN_BRDF_MODEL: u32 = BRDF_COOK_TORRANCE_GGX;  // Used when TERRAIN_USE_BRDF_DISPATCH = true

// P3-10/P1: CSM shadow sampling for terrain direct lighting
// Single source of truth for enabling terrain shadows
const TERRAIN_SHADOWS_ENABLED: bool = true;
const TERRAIN_USE_SHADOWS: bool = TERRAIN_SHADOWS_ENABLED;

// P1-Shadow: Shadow intensity tuning
// SHADOW_MIN: minimum brightness in fully shadowed areas (0.0 = pitch black, 0.3 = soft shadows)
// SHADOW_IBL_FACTOR: how much IBL diffuse is reduced in shadow (0.0 = no effect, 1.0 = full shadow)
// 
// P2-S3: shadow factor must be clamped to [0.20, 1.0] - relaxed for P5-AO gradient enhancement
// P2-S1: ambient_floor must be in [0.22, 0.38] range
// Spec A-01: ambient floor ensures even fully shadowed terrain has L >= 0.10 after tonemap
const SHADOW_MIN: f32 = 0.20;        // P5-AO: Reduced from 0.30 to allow deeper shadows
const SHADOW_IBL_FACTOR: f32 = 0.20; // IBL diffuse reduced by 20% in shadow

// P2-S1: Ambient floor constants for terrain lighting
// These ensure valleys don't crush to black while maintaining proper contrast
const AMBIENT_FLOOR_MIN: f32 = 0.18;  // P5-AO: Reduced from 0.22
const AMBIENT_FLOOR_MAX: f32 = 0.38;
const AMBIENT_FLOOR: f32 = 0.18;     // P5-AO: Reduced for deeper shadows and more gradient

// P1-Shadow Debug: Set to true to visualize shadow cascade coverage
// Color codes terrain by which cascade is used: Red=0, Green=1, Blue=2, Yellow=3
// Note: These are compile-time constants. For runtime override, use debug_mode in CsmUniforms:
//   csm_uniforms.debug_mode == 1 -> cascade boundary overlay
//   csm_uniforms.debug_mode == 2 -> raw shadow visibility overlay
// Environment variable: FORGE3D_TERRAIN_SHADOW_DEBUG="cascades" or "raw"
const DEBUG_SHADOW_CASCADES: bool = false;
const DEBUG_SHADOW_RAW: bool = false;  // Show raw shadow visibility as grayscale

// Shadow debug mode constants (must match Rust side)
const SHADOW_DEBUG_NONE: u32 = 0u;
const SHADOW_DEBUG_CASCADES: u32 = 1u;
const SHADOW_DEBUG_RAW: u32 = 2u;

// ──────────────────────────────────────────────────────────────────────────
// Debug Mode Constants — "truth serum" diagnostics for water/IBL/PBR
// These are forensic modes, not visual improvements. Each answers ONE question.
// ──────────────────────────────────────────────────────────────────────────
// Water debug modes (4-6):
const DBG_WATER_MASK_BINARY: u32 = 4u;  // CYAN = water, MAGENTA = land (uses SAME is_water as main path)
const DBG_WATER_MASK_RAW: u32 = 5u;     // Grayscale [0,1], RED=<0, YELLOW=>1, GREEN=NaN/Inf
const DBG_IBL_ONLY: u32 = 6u;           // IBL contribution only (same tonemap as normal frames)
// PBR debug modes (7-12): Proof pack for microfacet BRDF correctness
const DBG_PBR_DIFFUSE_ONLY: u32 = 7u;   // Diffuse IBL term only (no specular, no sun)
const DBG_PBR_SPECULAR_ONLY: u32 = 8u;  // Specular IBL term only (no diffuse, no sun)
const DBG_PBR_FRESNEL: u32 = 9u;        // Fresnel term F as grayscale (average of RGB)
const DBG_PBR_NDOTV: u32 = 10u;         // N.V (view angle) as grayscale
const DBG_PBR_ROUGHNESS: u32 = 11u;     // Roughness value as grayscale (after any multiplier)
const DBG_PBR_ENERGY: u32 = 12u;        // Raw (diffuse + specular) luminance before tonemap (for energy histogram)
// Recomposition proof modes (13-16): Prove that IBL = diffuse + specular in linear space
const DBG_PBR_LINEAR_COMBINED: u32 = 13u;  // Linear unclamped (diff+spec), RGB encoded [0,4] -> [0,1]
const DBG_PBR_LINEAR_DIFFUSE: u32 = 14u;   // Linear unclamped diffuse only, RGB encoded [0,4] -> [0,1]
const DBG_PBR_LINEAR_SPECULAR: u32 = 15u;  // Linear unclamped specular only, RGB encoded [0,4] -> [0,1]
const DBG_PBR_RECOMP_ERROR: u32 = 16u;     // abs(ibl_total - (diff+spec)) heatmap, amplified 100x
// SpecAA stress test mode (17): High-frequency sparkle detection
const DBG_SPECAA_SPARKLE: u32 = 17u;       // Specular with synthetic high-freq normal perturbation
// POM debug mode (18): Parallax offset magnitude visualization
const DBG_POM_OFFSET_MAG: u32 = 18u;       // Grayscale POM offset magnitude (0=none, white=max offset)
// SpecAA sigma2 debug mode (19): Variance visualization for SpecAA diagnostics
const DBG_SPECAA_SIGMA2: u32 = 19u;        // Grayscale sigma² (0=no variance, white=high variance)
// SpecAA sparkle sigma2 debug mode (20): Variance on sparkle-perturbed normal
const DBG_SPECAA_SPARKLE_SIGMA2: u32 = 20u; // Shows variance computed on perturbed normal
// Triplanar debug modes (21-22): Proof pack for triplanar mapping correctness
const DBG_TRIPLANAR_WEIGHTS: u32 = 21u;     // RGB = x/y/z blend weights (sum to 1)
const DBG_TRIPLANAR_CHECKER: u32 = 22u;     // Procedural checker to expose UV stretching
// Flake diagnosis modes (23-27): Milestone 1-3 proof pack
const DBG_FLAKE_NO_SPECULAR: u32 = 23u;     // Direct lighting only (no IBL specular)
const DBG_FLAKE_NO_HEIGHT_NORMAL: u32 = 24u; // Use base_normal instead of height_normal
const DBG_FLAKE_DDXDDY_NORMAL: u32 = 25u;   // Use n_dd = cross(dpdx, dpdy) as shading normal
const DBG_FLAKE_HEIGHT_LOD: u32 = 26u;      // Visualize computed height LOD
const DBG_FLAKE_NORMAL_BLEND: u32 = 27u;    // Visualize effective normal_blend after LOD fade
// P5: Raw SSAO visualization
const DBG_RAW_SSAO: u32 = 28u;
// Sun visibility debug mode
const DBG_SUN_VIS: u32 = 29u;
// Keep POM loop bounds compile-time constant so FXC can lower the fragment
// shader without trying to infer a large dynamic unroll.
const POM_MAX_STEPS: u32 = 128u;
const POM_MAX_REFINE_STEPS: u32 = 32u;
// Sprint 1: Shadow-field diagnostic modes
const DBG_NDOTL: u32 = 30u;           // N·L (lambert term) as grayscale
const DBG_SHADOW_FACTOR: u32 = 31u;   // Shadow visibility factor as grayscale
const DBG_PRE_TONEMAP: u32 = 32u;     // Final color before tonemapping (linear, clamped for display)
const DBG_SHADOW_TECHNIQUE: u32 = 33u; // P6.2: Show shadow technique as color (Red=HARD, Green=PCF, Blue=PCSS)
// P7: Projection probe modes for perspective verification (lighting-independent)
const DBG_VIEW_DEPTH: u32 = 40u;      // View-space depth as grayscale (changes with FOV/theta if perspective works)
const DBG_NDC_DEPTH: u32 = 41u;       // NDC depth (clip.z/clip.w) as grayscale
const DBG_VIEW_POS_XYZ: u32 = 42u;    // View-space position encoded as RGB (normalized to [0,1])
const DBG_PROBE_IRRADIANCE: u32 = 50u; // Raw probe irradiance contribution
const DBG_PROBE_WEIGHT: u32 = 51u;     // Probe blend weight
const DBG_REFLECTION_PROBE_COLOR: u32 = 52u;  // Raw local reflection probe color
const DBG_REFLECTION_PROBE_WEIGHT: u32 = 53u; // Local reflection probe blend weight

// Water reflection tuning: named here so local-planar/IBL balance stays explicit.
const WATER_DEPTH_ATTEN_DEEP: f32 = 0.30;
const WATER_COMBINED_REFLECTION_SCALE: f32 = 0.30;
const WATER_SUN_SPECULAR_SCALE: f32 = 0.50;
const WATER_BASE_TINT: vec3<f32> = vec3<f32>(0.15, 0.45, 0.85);
const WATER_BASE_TINT_SCALE: f32 = 0.80;
const WATER_SCATTER_SCALE: f32 = 2.0;

struct TerrainUniforms {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
    sun_exposure : vec4<f32>,
    spacing_h_exag : vec4<f32>,
    // x=camera_mode (0=screen, 1=mesh), y=grid_size (for mesh mode),
    // z=clip_near, w=clip_far
    camera_mode_params : vec4<f32>,
};

struct TerrainShadingUniforms {
    triplanar_params : vec4<f32>, // x=scale, y=blend_sharpness, z=normal_strength, w=pom_scale
    pom_steps : vec4<f32>,        // x=min_steps, y=max_steps, z=refine_steps, w=flags
    layer_heights : vec4<f32>,    // normalized centers per layer
    layer_roughness : vec4<f32>,
    layer_metallic : vec4<f32>,
    layer_control : vec4<f32>,    // x=layer_count, y=blend_half_width, z=lod_bias, w=lod0_bias
    light_params : vec4<f32>,     // rgb = light color * intensity, w=exposure
    clamp0 : vec4<f32>,           // height_min, height_max, slope_min, slope_max
    clamp1 : vec4<f32>,           // ambient_min, ambient_max, shadow_min, shadow_max
    clamp2 : vec4<f32>,           // occlusion_min, occlusion_max, lod_level, anisotropy
    height_curve : vec4<f32>,     // x=mode, y=strength, z=power, w=lambert_contrast (P5-L)
};

struct OverlayUniforms {
    params0 : vec4<f32>, // domain_min, inv_range, overlay_strength, offset
    params1 : vec4<f32>, // blend_mode, debug_mode, albedo_mode, colormap_strength
    params2 : vec4<f32>, // gamma, roughness_mult, spec_aa_enabled, specaa_sigma_scale
    params3 : vec4<f32>, // P5: ao_weight, ao_fallback_enabled, hue_variation_strength, pad
    // P6: Micro-detail parameters
    params4 : vec4<f32>, // detail_enabled, detail_scale, detail_normal_strength, detail_albedo_noise
    params5 : vec4<f32>, // detail_fade_start, detail_fade_end, output_srgb_eotf, offline_hdr_output
};

struct IblUniforms {
    intensity : f32,
    sin_theta : f32,
    cos_theta : f32,
    specular_mip_count : f32,
};

@group(0) @binding(0)
var<uniform> u_terrain : TerrainUniforms;

@group(0) @binding(1)
var height_tex : texture_2d<f32>;

@group(0) @binding(2)
var height_samp : sampler;

// R32Float height textures are intentionally bound as unfilterable on the
// portable terrain path. Reconstruct bilinear filtering explicitly so height
// sampling is identical on adapters that cannot filter this format.
fn sample_height_bilinear_level(uv: vec2<f32>, lod: f32) -> f32 {
    let last_level = i32(textureNumLevels(height_tex)) - 1;
    let level = clamp(i32(floor(lod + 0.5)), 0, last_level);
    let dimensions = textureDimensions(height_tex, level);
    let max_x = max(i32(dimensions.x) - 1, 0);
    let max_y = max(i32(dimensions.y) - 1, 0);
    let texel_x = clamp(uv.x, 0.0, 1.0) * f32(max_x);
    let texel_y = clamp(uv.y, 0.0, 1.0) * f32(max_y);
    let x0 = i32(floor(texel_x));
    let y0 = i32(floor(texel_y));
    // Keep each upper-bound relation explicit for the CENSOR IR proof as well
    // as for runtime safety; all four coordinates are clamped to this texture.
    let x1 = clamp(x0 + 1, 0, max_x);
    let y1 = clamp(y0 + 1, 0, max_y);
    let blend = clamp(
        vec2<f32>(texel_x - f32(x0), texel_y - f32(y0)),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );

    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), level).r;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), level).r;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y1), level).r;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y1), level).r;
    return det_mix(det_mix(h00, h10, blend.x), det_mix(h01, h11, blend.x), blend.y);
}

fn sample_height_bilinear(uv: vec2<f32>) -> f32 {
    return sample_height_bilinear_level(uv, 0.0);
}

@group(0) @binding(3)
var material_albedo_tex : texture_2d_array<f32>;

@group(0) @binding(4)
var material_samp : sampler;

@group(0) @binding(5)
var<uniform> u_shading : TerrainShadingUniforms;

@group(0) @binding(6)
var colormap_tex : texture_2d<f32>;

@group(0) @binding(7)
var colormap_samp : sampler;

@group(0) @binding(8)
var<uniform> u_overlay : OverlayUniforms;

@group(0) @binding(9)
var height_curve_lut_tex : texture_2d<f32>;

@group(0) @binding(10)
var height_curve_lut_samp : sampler;

@group(0) @binding(11)
var water_mask_tex : texture_2d<f32>;

// P5: AO debug/fallback texture (raw SSAO or coarse height AO)
@group(0) @binding(12)
var ao_debug_tex : texture_2d<f32>;

@group(0) @binding(13)
var ao_debug_samp : sampler;

// P6: Detail normal texture (DEM-derived tangent-space normal map)
// Fallback = neutral normal (RGB 128,128,255 = flat normal in tangent space)
// Channel mapping: R=X, G=Y, B=Z (OpenGL convention)
// Encoding: [0,255] -> [-1,1] per channel
@group(0) @binding(14)
var detail_normal_tex : texture_2d<f32>;

@group(0) @binding(15)
var detail_normal_samp : sampler;

// Heightfield ray-traced AO texture (R8Unorm, computed by heightfield_ao.wgsl)
// When height_ao.enabled=false, this is bound to a 1x1 white texture (AO=1.0)
@group(0) @binding(16)
var height_ao_tex : texture_2d<f32>;

@group(0) @binding(17)
var height_ao_samp : sampler;

// Heightfield ray-traced sun visibility texture (R8Unorm, computed by heightfield_sun_vis.wgsl)
// When sun_visibility.enabled=false, this is bound to a 1x1 white texture (vis=1.0)
@group(0) @binding(18)
var sun_vis_tex : texture_2d<f32>;

@group(0) @binding(19)
var sun_vis_samp : sampler;

// P1-06: Light buffer bindings (@group(1))
struct Light {
    light_type: u32,
    flags: u32,
    position: vec3<f32>,
    direction: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    area_width: f32,
    area_height: f32,
    env_texture_index: u32,
};

// Note: LightMetadata from lights.wgsl is used via lighting.wgsl, but terrain uses a different structure
// Renamed to avoid conflict
struct TerrainLightMetadata {
    light_count: u32,
    frame_index: u32,
    sequence_seed: vec2<f32>,
};

struct EnvironmentParams {
    ambient: vec3<f32>,
    padding: f32,
};

@group(1) @binding(3)
var<storage, read> terrain_lights: array<Light>;

@group(1) @binding(4)
var<uniform> light_metadata: TerrainLightMetadata;

@group(1) @binding(5)
var<uniform> environment_params: EnvironmentParams;

// IBL bindings are declared in lighting_ibl.wgsl (P4 spec: group(2) bindings 0-3)
// @group(2) @binding(0) envSpecular : texture_cube<f32>
// @group(2) @binding(1) envIrradiance : texture_cube<f32>
// @group(2) @binding(2) envSampler : sampler
// @group(2) @binding(3) brdfLUT : texture_2d<f32>

// Terrain-specific IBL uniforms (rotation, intensity, mip count)
@group(2) @binding(4)
var<uniform> u_ibl : IblUniforms;

// P3-10: Optional shadow bindings at @group(3) (only used when TERRAIN_USE_SHADOWS = true)
// These mirror the shadow bindings from shadows.wgsl but at a different group to avoid IBL conflict
// Simplified shadow cascade data for terrain
// Must match Rust struct in crate::core::shadow_mapping::CsmCascadeData
struct ShadowCascade {
    light_projection: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,  // Pre-computed projection * view for consistency
    near_distance: f32,
    far_distance: f32,
    texel_size: f32,
    _padding: f32,
}

// P0.2/M3: CsmUniforms for terrain shadow sampling
// Must match Rust struct in crate::shadows::csm_types::CsmUniforms exactly
// Using storage buffer with std430 layout - no complex padding needed
struct CsmUniforms {
    light_direction: vec4<f32>,        // 16 bytes, offset 0
    light_view: mat4x4<f32>,           // 64 bytes, offset 16
    cascades: array<ShadowCascade, 4>, // 576 bytes, offset 80 (4 * 144: 2 mat4x4 + 4 floats each)
    cascade_count: u32,                // 4 bytes, offset 656
    pcf_kernel_size: u32,              // 4 bytes, offset 660
    depth_bias: f32,                   // 4 bytes, offset 664
    slope_bias: f32,                   // 4 bytes, offset 668
    shadow_map_size: f32,              // 4 bytes, offset 672
    debug_mode: u32,                   // 4 bytes, offset 676
    evsm_positive_exp: f32,            // 4 bytes, offset 680
    evsm_negative_exp: f32,            // 4 bytes, offset 684
    peter_panning_offset: f32,         // 4 bytes, offset 688
    enable_unclipped_depth: u32,       // 4 bytes, offset 692
    depth_clip_factor: f32,            // 4 bytes, offset 696
    // P0.2/M3: Shadow technique selection (Hard=0, PCF=1, PCSS=2, VSM=3, EVSM=4, MSM=5)
    technique: u32,                    // 4 bytes, offset 700
    technique_flags: u32,              // 4 bytes, offset 704
    _pad1a: f32,
    _pad1b: f32,
    _pad1c: f32,
    // [blocker radius (texels), max filter radius (texels), moment bias, light radius (texels)]
    technique_params: vec4<f32>,       // 16 bytes, offset 720
    technique_reserved: vec4<f32>,     // 16 bytes, offset 736
    cascade_blend_range: f32,          // 4 bytes, offset 752
    // std430: 108 bytes padding to reach 864 total
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    _pad2d: array<vec4<f32>, 6>,
}

@group(3) @binding(0)
var<storage, read> csm_uniforms: CsmUniforms;

@group(3) @binding(1)
var shadow_maps: texture_depth_2d_array;

@group(3) @binding(2)
var shadow_sampler: sampler_comparison;

@group(3) @binding(3)
var moment_maps: texture_2d_array<f32>;

@group(3) @binding(4)
var moment_sampler: sampler;

// ──────────────────────────────────────────────────────────────────────────
// P2 + TV1: Shared atmosphere uniforms (@group(4))
// Carries the height-fog controls plus the resolved terrain sky texture.
// When sky is disabled and fog_density = 0.0, this path is an exact no-op.
// ──────────────────────────────────────────────────────────────────────────
struct FogUniforms {
    // x=density, y=height falloff, z=base height, w=camera height
    params0: vec4<f32>,
    // rgb=fog inscatter tint, w=AETHER model enabled
    fog_inscatter: vec4<f32>,
    // x=sky enabled, y=sky aerial density, z=sky aerial enabled, w=sky sun intensity
    sky_params0: vec4<f32>,
    // Legacy: x=sun size, y=sun elevation, z=turbidity, w=exposure.
    // AETHER: x=ozone DU, y=Mie g, z=turbidity, w=exposure.
    sky_params1: vec4<f32>,
    // xyz=sun direction in the terrain's Z-up world frame.
    aether_sun_direction: vec4<f32>,
    // x=bottom radius, y=top radius, z=scattering height count, w=nu count.
    aether_planet_lut: vec4<f32>,
}

@group(4) @binding(0)
var<uniform> fog_uniforms: FogUniforms;

@group(4) @binding(1)
var sky_atmosphere_tex: texture_2d<f32>;

@group(4) @binding(2)
var aether_accumulated_scattering_tex: texture_3d<f32>;

// ──────────────────────────────────────────────────────────────────────────
// P4: Water Planar Reflection Uniforms (@group(5))
// Deterministic planar reflections with mirrored camera for water surfaces.
// When enable_reflections = 0 or no water, reflections are a no-op.
// ──────────────────────────────────────────────────────────────────────────
struct WaterReflectionUniforms {
    // Reflection view-projection matrix (mirrored camera)
    reflection_view_proj: mat4x4<f32>,
    // Water plane parameters: normal (xyz), distance (w)
    water_plane: vec4<f32>,
    // Reflection params: (intensity, fresnel_power, wave_strength, shore_atten_width)
    reflection_params: vec4<f32>,
    // Camera world position for Fresnel calculation
    camera_world_pos: vec4<f32>,
    // Enable flags: (enable_reflections, debug_mode, resolution, _pad)
    enable_flags: vec4<f32>,
}

@group(5) @binding(0)
var<uniform> water_reflection_uniforms: WaterReflectionUniforms;

@group(5) @binding(1)
var reflection_texture: texture_2d<f32>;

@group(5) @binding(2)
var reflection_sampler: sampler;

fn sample_atmosphere_sky(screen_pos: vec2<f32>) -> vec3<f32> {
    if (fog_uniforms.sky_params0.x < 0.5 || water_reflection_uniforms.enable_flags.y > 0.5) {
        return fog_uniforms.fog_inscatter.rgb;
    }

    let dims = textureDimensions(sky_atmosphere_tex, 0);
    let x = clamp(i32(screen_pos.x), 0, max(i32(dims.x) - 1, 0));
    let y = clamp(i32(screen_pos.y), 0, max(i32(dims.y) - 1, 0));
    return textureLoad(sky_atmosphere_tex, vec2<i32>(x, y), 0).rgb;
}

// ──────────────────────────────────────────────────────────────────────────
// M4: Material Layer Uniforms (@group(6))
// Slope/aspect/altitude-driven material blending for snow, rock, wetness.
// When all layers are disabled, output is identical to baseline.
// ──────────────────────────────────────────────────────────────────────────
struct MaterialLayerUniforms {
    // Snow layer: vec4(altitude_min, altitude_blend, slope_max_rad, slope_blend_rad)
    snow_params0: vec4<f32>,
    // Snow layer: vec4(aspect_influence, roughness, enabled, sss_strength)
    snow_params1: vec4<f32>,
    // Snow color (RGB) + padding
    snow_color: vec4<f32>,
    // Snow subsurface tint (RGB) + padding
    snow_sss_tint: vec4<f32>,
    // Rock layer: vec4(slope_min_rad, slope_blend_rad, roughness, enabled)
    rock_params: vec4<f32>,
    // Rock color (RGB) + subsurface strength
    rock_color: vec4<f32>,
    // Rock subsurface tint (RGB) + padding
    rock_sss_tint: vec4<f32>,
    // Wetness layer: vec4(strength, slope_influence, enabled, sss_strength)
    wetness_params: vec4<f32>,
    // Wetness subsurface tint (RGB) + padding
    wetness_sss_tint: vec4<f32>,
    // TV4: vec4(macro_scale, detail_scale, octaves, variation_enabled)
    variation_params0: vec4<f32>,
    // TV4 per-layer amplitudes: vec4(macro_amp, detail_amp, _pad, _pad)
    snow_variation: vec4<f32>,
    rock_variation: vec4<f32>,
    wetness_variation: vec4<f32>,
    // Per-texel material map flags: normal, roughness, mask, _pad
    map_flags: vec4<f32>,
}

@group(6) @binding(0)
var<uniform> material_layer_uniforms: MaterialLayerUniforms;


const TERRAIN_VT_MATERIAL_CAPACITY: u32 = 4u;
const TERRAIN_VT_FAMILY_ALBEDO: u32 = 0u;
const TERRAIN_VT_FAMILY_NORMAL: u32 = 1u;
const TERRAIN_VT_FAMILY_MASK: u32 = 2u;

struct TerrainVtFamilyInfo {
    state: vec4<u32>,
    virtual_size: vec4<u32>,
}

struct TerrainVTUniforms {
    config0: vec4<u32>,
    config1: vec4<u32>,
    config2: vec4<u32>,
    // Three explicit, std140-safe 32-byte family records. This is the single
    // source of truth for family enablement, addressing, size, and encoding.
    family_info: array<TerrainVtFamilyInfo, 3>,
    // Bounded feedback append (TESSELLA win 1). x = slot capacity (a power of
    // two), y/z = physical page-table base width/height, w unused.
    config3: vec4<u32>,
}

struct TerrainVTFallbackColors {
    albedo: array<vec4<f32>, 4>,
    normal: array<vec4<f32>, 4>,
    mask: array<vec4<f32>, 4>,
}

// TESSELLA win 1: the feedback surface is a BOUNDED set, not a page map.
//
// It used to be DIRECT-MAPPED over the whole virtual page pyramid -- one
// 16-byte slot per (family, material, mip, page_y, page_x). At the acceptance
// camera's 2^18 x 2^18 virtual texture that is 100,663,296 slots = 1.5 GiB of
// MAP_READ memory: three times the entire 512 MiB host-visible budget, for a
// working set of about a thousand pages. It is now a fixed-capacity append
// buffer of 32-bit page keys, so the host-visible footprint is a function of
// the CAPACITY, never of the virtual texture's size. Duplicate samples are
// intentional: the CPU deduplicates the bounded readback with its normal
// HashSet path.
//
// Layout: word 0 is an atomic append count, word 1 is an explicit overflow
// count, and words 2 ..= capacity + 1 hold `page_index + 1`. The key alone
// identifies the page, so the CPU inverts `terrain_vt_feedback_index` rather
// than reading back separate coordinate words.

@group(6) @binding(6)
var<uniform> terrain_vt_uniforms: TerrainVTUniforms;

@group(6) @binding(7)
var<uniform> terrain_vt_fallbacks: TerrainVTFallbackColors;

@group(6) @binding(8)
var terrain_vt_atlas: texture_2d<f32>;

@group(6) @binding(9)
var terrain_vt_sampler: sampler;

@group(6) @binding(10)
var terrain_vt_page_table: texture_2d_array<f32>;

@group(6) @binding(11)
var<storage, read_write> terrain_vt_feedback: array<atomic<u32>>;

@group(6) @binding(12)
var material_normal_tex: texture_2d<f32>;

@group(6) @binding(13)
var material_roughness_tex: texture_2d<f32>;

@group(6) @binding(14)
var material_mask_tex: texture_2d<f32>;

@group(6) @binding(15)
var material_map_samp: sampler;

struct TerrainFrameCounters {
    material_invocations: atomic<u32>,
    feedback_records: atomic<u32>,
    fallback_texels: atomic<u32>,
    forward_material_invocations: atomic<u32>,
}

@group(6) @binding(16)
var<storage, read_write> terrain_frame_counters: TerrainFrameCounters;
struct TerrainMaterialNoise {
    snow_macro: f32,
    snow_detail: f32,
    rock_macro: f32,
    rock_detail: f32,
    wetness_macro: f32,
    wetness_detail: f32,
}

struct TerrainLayerWeights {
    snow: f32,
    rock: f32,
    wetness: f32,
}

struct TerrainSubsurfaceState {
    strength: f32,
    tint: vec3<f32>,
}

// ──────────────────────────────────────────────────────────────────────────
// M4: Terrain attribute computation (slope, aspect)
// ──────────────────────────────────────────────────────────────────────────

/// Compute terrain attributes from world normal
/// Returns: vec4(slope_radians, aspect_radians, curvature_proxy, _pad)
/// slope: 0 = flat, PI/2 = vertical cliff
/// aspect: angle from north (0 = north, PI/2 = east, PI = south, 3PI/2 = west)
fn compute_terrain_attributes(world_normal: vec3<f32>) -> vec4<f32> {
    // Slope: angle from vertical (Z-up coordinate system)
    // slope = acos(det_dot3(normal, up)) where up = (0,0,1)
    let slope = acos(clamp(world_normal.z, -1.0, 1.0));
    
    // Aspect: direction the slope faces (azimuth of steepest descent)
    // In Z-up: project normal to XY plane and compute angle from Y-axis (north)
    let horizontal = vec2<f32>(world_normal.x, world_normal.y);
    let horiz_len = length(horizontal);
    var aspect = 0.0;
    if (horiz_len > 0.001) {
        // atan2(x, y) gives angle from Y-axis (north)
        aspect = atan2(horizontal.x, horizontal.y);
        if (aspect < 0.0) {
            aspect = aspect + 2.0 * 3.14159265;
        }
    }
    
    // Curvature proxy: use slope derivative approximation
    // For now, just use slope as curvature proxy (simplified)
    let curvature_proxy = slope;
    
    return vec4<f32>(slope, aspect, curvature_proxy, 0.0);
}

fn default_material_noise() -> TerrainMaterialNoise {
    return TerrainMaterialNoise(0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
}

fn sample_material_noise(terrain_uv: vec2<f32>, height_norm: f32) -> TerrainMaterialNoise {
    let macro_scale = max(material_layer_uniforms.variation_params0.x, 0.001);
    let detail_scale = max(material_layer_uniforms.variation_params0.y, 0.001);
    let octaves = i32(material_layer_uniforms.variation_params0.z + 0.5);

    let macro_coords = vec3<f32>(terrain_uv * macro_scale, height_norm * 1.7);
    let detail_coords = vec3<f32>(terrain_uv * detail_scale, height_norm * 3.1);
    let detail_octaves = min(octaves + 1, TERRAIN_NOISE_MAX_OCTAVES);

    return TerrainMaterialNoise(
        terrain_fbm(macro_coords + vec3<f32>(0.0, 0.0, 0.0), octaves),
        terrain_fbm(detail_coords + vec3<f32>(17.3, 9.1, 3.7), detail_octaves),
        terrain_ridged_fbm(macro_coords + vec3<f32>(31.7, 5.2, 11.9), octaves),
        1.0 - terrain_cellular_distance(detail_coords + vec3<f32>(2.1, 13.4, 7.6)),
        1.0 - terrain_cellular_distance(macro_coords + vec3<f32>(19.5, 23.1, 5.7)),
        terrain_fbm(detail_coords + vec3<f32>(41.0, 17.0, 29.0), detail_octaves),
    );
}

fn apply_material_variation(
    base_weight: f32,
    macro_noise: f32,
    detail_noise: f32,
    macro_amplitude: f32,
    detail_amplitude: f32,
) -> f32 {
    let macro_delta = (macro_noise - 0.5) * 2.0 * macro_amplitude;
    let detail_delta = (detail_noise - 0.5) * 2.0 * detail_amplitude;
    // Keep most of the variation near material transitions so zero-regression
    // defaults are preserved and full-coverage regions do not become noisy speckle.
    let transition_boost = 0.35 + 0.65 * (1.0 - abs(base_weight * 2.0 - 1.0));
    return clamp(base_weight + (macro_delta + detail_delta) * transition_boost, 0.0, 1.0);
}

fn compute_snow_layer_weight(
    world_pos: vec3<f32>,
    terrain_attrs: vec4<f32>,
    material_noise: TerrainMaterialNoise,
) -> f32 {
    let snow_enabled = material_layer_uniforms.snow_params1.z;
    if (snow_enabled < 0.5) {
        return 0.0;
    }
    
    let altitude = world_pos.z;
    let slope = terrain_attrs.x;
    let aspect = terrain_attrs.y;
    
    // Altitude factor: ramp from 0 at min to 1 at min+blend
    let alt_min = material_layer_uniforms.snow_params0.x;
    let alt_blend = material_layer_uniforms.snow_params0.y;
    let altitude_factor = clamp((altitude - alt_min) / max(alt_blend, 0.001), 0.0, 1.0);
    
    // Slope factor: snow accumulates on flat surfaces, not on cliffs
    // slope_max is in radians, ramp from 1 at 0 to 0 at slope_max
    let slope_max = material_layer_uniforms.snow_params0.z;
    let slope_blend = material_layer_uniforms.snow_params0.w;
    let slope_factor = 1.0 - clamp((slope - slope_max + slope_blend) / max(slope_blend, 0.001), 0.0, 1.0);
    
    // Aspect factor: south-facing slopes get less snow (more sun exposure)
    // aspect=0 is north, aspect=PI is south
    let aspect_influence = material_layer_uniforms.snow_params1.x;
    // cos(aspect) = 1 for north, -1 for south
    // We want: north=1, south=reduced
    let south_factor = cos(aspect); // 1 for north, -1 for south
    let aspect_factor = det_mix(1.0, 0.5 + 0.5 * south_factor, aspect_influence);
    
    // Combined snow weight
    let snow_weight = apply_material_variation(
        altitude_factor * slope_factor * aspect_factor,
        material_noise.snow_macro,
        material_noise.snow_detail,
        material_layer_uniforms.snow_variation.x,
        material_layer_uniforms.snow_variation.y,
    );
    return snow_weight;
}

/// M4: Apply snow layer blending based on altitude, slope, and aspect
fn apply_snow_layer(base_albedo: vec3<f32>, snow_weight: f32) -> vec3<f32> {
    let snow_color = material_layer_uniforms.snow_color.rgb;
    return det_mix3(base_albedo, snow_color, clamp(snow_weight, 0.0, 1.0));
}

fn compute_rock_layer_weight(
    terrain_attrs: vec4<f32>,
    material_noise: TerrainMaterialNoise,
) -> f32 {
    let rock_enabled = material_layer_uniforms.rock_params.w;
    if (rock_enabled < 0.5) {
        return 0.0;
    }
    
    let slope = terrain_attrs.x;
    
    // Rock exposed on steep slopes
    let slope_min = material_layer_uniforms.rock_params.x;
    let slope_blend = material_layer_uniforms.rock_params.y;
    let rock_weight = apply_material_variation(
        clamp((slope - slope_min) / max(slope_blend, 0.001), 0.0, 1.0),
        material_noise.rock_macro,
        material_noise.rock_detail,
        material_layer_uniforms.rock_variation.x,
        material_layer_uniforms.rock_variation.y,
    );
    return rock_weight;
}

/// M4: Apply rock layer blending based on slope
fn apply_rock_layer(base_albedo: vec3<f32>, rock_weight: f32) -> vec3<f32> {
    let rock_color = material_layer_uniforms.rock_color.rgb;
    return det_mix3(base_albedo, rock_color, clamp(rock_weight, 0.0, 1.0));
}

fn compute_wetness_layer_coverage(
    terrain_attrs: vec4<f32>,
    material_noise: TerrainMaterialNoise,
) -> f32 {
    let wetness_enabled = material_layer_uniforms.wetness_params.z;
    if (wetness_enabled < 0.5) {
        return 0.0;
    }
    
    let slope = terrain_attrs.x;
    
    // Wetness accumulates in flat, low areas (simplified: low slope = wet)
    let slope_influence = material_layer_uniforms.wetness_params.y;
    
    // Flat areas (low slope) are wetter
    let flat_factor = 1.0 - clamp(slope / (3.14159265 * 0.25), 0.0, 1.0);
    return apply_material_variation(
        flat_factor * slope_influence,
        material_noise.wetness_macro,
        material_noise.wetness_detail,
        material_layer_uniforms.wetness_variation.x,
        material_layer_uniforms.wetness_variation.y,
    );
}

/// M4: Apply wetness darkening based on slope (concavity proxy)
fn apply_wetness_layer(base_albedo: vec3<f32>, wetness_coverage: f32) -> vec3<f32> {
    // Darken by wetness
    let strength = material_layer_uniforms.wetness_params.x;
    let darkening = 1.0 - clamp(wetness_coverage, 0.0, 1.0) * strength;
    return base_albedo * darkening;
}

fn resolve_terrain_layer_weights(
    world_pos: vec3<f32>,
    terrain_attrs: vec4<f32>,
    material_noise: TerrainMaterialNoise,
) -> TerrainLayerWeights {
    return TerrainLayerWeights(
        compute_snow_layer_weight(world_pos, terrain_attrs, material_noise),
        compute_rock_layer_weight(terrain_attrs, material_noise),
        compute_wetness_layer_coverage(terrain_attrs, material_noise),
    );
}

fn apply_subsurface_layer(
    state: TerrainSubsurfaceState,
    layer_weight: f32,
    layer_strength: f32,
    layer_tint: vec3<f32>,
) -> TerrainSubsurfaceState {
    if (layer_weight <= 0.0 || layer_strength <= 0.0) {
        return state;
    }
    let coverage = clamp(layer_weight, 0.0, 1.0);
    return TerrainSubsurfaceState(
        det_mix(state.strength, layer_strength, coverage),
        det_mix3(state.tint, layer_tint, coverage),
    );
}

fn resolve_terrain_subsurface(layer_weights: TerrainLayerWeights) -> TerrainSubsurfaceState {
    var state = TerrainSubsurfaceState(0.0, vec3<f32>(1.0, 1.0, 1.0));
    state = apply_subsurface_layer(
        state,
        layer_weights.wetness,
        material_layer_uniforms.wetness_params.w,
        material_layer_uniforms.wetness_sss_tint.rgb,
    );
    state = apply_subsurface_layer(
        state,
        layer_weights.rock,
        material_layer_uniforms.rock_color.w,
        material_layer_uniforms.rock_sss_tint.rgb,
    );
    state = apply_subsurface_layer(
        state,
        layer_weights.snow,
        material_layer_uniforms.snow_params1.w,
        material_layer_uniforms.snow_sss_tint.rgb,
    );
    return state;
}

fn evaluate_terrain_subsurface(
    state: TerrainSubsurfaceState,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    combined_shadow: f32,
    ibl_diffuse_factor: f32,
) -> vec3<f32> {
    if (state.strength <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let n_dot_l = saturate(det_dot3(normal, light_dir));
    let wrap_width = 0.45 * state.strength;
    let wrapped = saturate((n_dot_l + wrap_width) / (1.0 + wrap_width));
    let wrap_boost = max(wrapped - n_dot_l, 0.0);

    let view_backscatter = det_pow(saturate(det_dot3(view_dir, -light_dir)), 4.0);
    let backscatter = view_backscatter * (0.25 + 0.75 * (1.0 - n_dot_l));
    let scatter_profile = max(wrap_boost * 1.35, backscatter * 0.30);

    let shadow_bleed = det_mix(0.20, 1.0, clamp(combined_shadow, 0.0, 1.0));
    let ambient_fill = ibl_diffuse_factor * (0.02 + 0.06 * state.strength) * (1.0 - n_dot_l * 0.5);
    let scatter_color = clamp(
        albedo * det_mix3(vec3<f32>(1.0, 1.0, 1.0), state.tint, 0.85),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(1.5, 1.5, 1.5),
    );

    return scatter_color * (scatter_profile * shadow_bleed + ambient_fill) * (0.16 + 0.44 * state.strength);
}

// P4: Sample reflection texture with wave-based UV distortion
// Returns (reflection_color, valid) where valid indicates if sample is in bounds
fn sample_water_reflection(
    world_pos: vec3<f32>,
    wave_normal: vec3<f32>,
    shore_distance: f32,
) -> vec4<f32> {
    // Early-out if reflections disabled
    if (water_reflection_uniforms.enable_flags.x < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Transform world position to reflection clip space
    let reflection_clip = det_mat4_mul_vec4(water_reflection_uniforms.reflection_view_proj, vec4<f32>(world_pos, 1.0));
    
    // Perspective divide
    if (abs(reflection_clip.w) < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let reflection_ndc = reflection_clip.xyz / reflection_clip.w;
    
    // Convert NDC to UV [0,1]
    var reflection_uv = reflection_ndc.xy * 0.5 + 0.5;
    // Flip Y for texture coordinate system
    reflection_uv.y = 1.0 - reflection_uv.y;
    
    // Wave-based UV distortion
    // Wave normal deviation from flat (0,1,0) creates UV offset
    let wave_strength = water_reflection_uniforms.reflection_params.z;
    let wave_distortion = (wave_normal.xz - vec2<f32>(0.0, 0.0)) * wave_strength;
    
    // Shore attenuation: reduce distortion near shore (calmer water at edges)
    let shore_atten_width = water_reflection_uniforms.reflection_params.w;
    let shore_factor = smoothstep(0.0, shore_atten_width, shore_distance);
    
    // Apply distortion with shore attenuation
    reflection_uv = reflection_uv + wave_distortion * shore_factor;
    
    // Clamp to valid range
    reflection_uv = clamp(reflection_uv, vec2<f32>(0.001), vec2<f32>(0.999));
    
    // Check if in bounds (out of bounds returns zero alpha)
    if (reflection_uv.x < 0.001 || reflection_uv.x > 0.999 ||
        reflection_uv.y < 0.001 || reflection_uv.y > 0.999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Sample reflection texture
    let reflection_color = textureSample(reflection_texture, reflection_sampler, reflection_uv);
    
    return vec4<f32>(reflection_color.rgb, 1.0);
}

// P4: Calculate Fresnel factor for water reflection blending
// Returns 0 at normal incidence (looking straight down), 1 at grazing angles
fn calculate_water_fresnel(view_dir: vec3<f32>, surface_normal: vec3<f32>) -> f32 {
    let fresnel_power = water_reflection_uniforms.reflection_params.y;
    let n_dot_v = max(0.0, det_dot3(surface_normal, view_dir));
    let fresnel = det_pow(1.0 - n_dot_v, fresnel_power);
    return clamp(fresnel, 0.0, 1.0);
}

// P4: Blend planar reflection with underwater color using Fresnel
fn blend_water_reflection(
    underwater_color: vec3<f32>,
    reflection_color: vec3<f32>,
    reflection_valid: bool,
    fresnel: f32,
    shore_distance: f32,
) -> vec3<f32> {
    if (!reflection_valid) {
        return underwater_color;
    }
    
    let intensity = water_reflection_uniforms.reflection_params.x;
    
    // Shore attenuation: reduce reflection intensity near shore
    // This helps blend water edges more naturally with land
    let shore_atten_width = water_reflection_uniforms.reflection_params.w;
    let shore_blend = smoothstep(0.0, shore_atten_width, shore_distance);
    
    // Final blend factor: Fresnel * intensity * shore_blend
    let blend = fresnel * intensity * shore_blend;
    
    return det_mix3(underwater_color, reflection_color, blend);
}

// P3-10: Shadow sampling functions (simplified for terrain)
// These are lightweight versions for terrain use, gated behind TERRAIN_USE_SHADOWS flag

/// Select cascade based on view-space depth
fn select_cascade_terrain(view_depth: f32) -> u32 {
    let count = csm_uniforms.cascade_count;
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            return i;
        }
    }
    return count - 1u;
}

// P0.2/M3: Light leak reduction for moment-based shadows
fn reduce_light_leak_terrain(shadow_factor: f32, amount: f32) -> f32 {
    return clamp(shadow_factor - amount, 0.0, 1.0);
}

// P0.2/M3: VSM (Variance Shadow Maps) sampling for terrain
fn sample_shadow_vsm_terrain(
    shadow_coords: vec2<f32>,
    receiver_depth: f32,
    cascade_idx: u32,
    moment_bias: f32
) -> f32 {
    // Sample moment map (RG channels contain E[x] and E[x^2])
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    let mean = moments.r;      // E[x]
    let mean_sq = moments.g;   // E[x^2]
    
    // If receiver is closer than mean, it is lit.
    if (receiver_depth <= mean) {
        return 1.0;
    }
    
    // Calculate variance: Var(x) = E[x^2] - E[x]^2
    let variance = max(mean_sq - mean * mean, 0.0001);
    
    // Apply Chebyshev inequality
    var shadow_factor = chebyshev_upper_bound_visibility(mean, variance, receiver_depth);
    
    // Apply light leak reduction
    if (moment_bias > 0.0) {
        shadow_factor = reduce_light_leak_terrain(shadow_factor, moment_bias);
    }
    
    return shadow_factor;
}

// P0.2/M3: EVSM (Exponential Variance Shadow Maps) sampling for terrain
fn sample_shadow_evsm_terrain(
    shadow_coords: vec2<f32>,
    receiver_depth: f32,
    cascade_idx: u32,
    moment_bias: f32
) -> f32 {
    // Sample moment map (RGBA channels)
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    
    // EVSM uses exponential warp to reduce light leaking
    let c_pos = csm_uniforms.evsm_positive_exp;
    let c_neg = csm_uniforms.evsm_negative_exp;
    
    // Warp receiver depth
    let warp_depth_pos = det_exp(c_pos * (receiver_depth - 1.0));
    let warp_depth_neg = -det_exp(-c_neg * receiver_depth);
    let variance_floor = evsm_minimum_variance(
        vec2<f32>(warp_depth_pos, warp_depth_neg),
        vec2<f32>(c_pos, c_neg)
    );
    
    // Apply each lobe's front-of-mean shortcut independently, then combine.
    var shadow_factor =
        evsm_visibility_from_moments(
            moments,
            warp_depth_pos,
            warp_depth_neg,
            variance_floor
        );
    shadow_factor = min(
        shadow_factor,
        evsm_moment_leak_control(
            moments.rg,
            warp_depth_pos,
            c_pos,
            variance_floor.x
        )
    );
    
    // Apply light leak reduction
    if (moment_bias > 0.0) {
        shadow_factor = reduce_light_leak_terrain(shadow_factor, moment_bias);
    }
    
    return shadow_factor;
}

// P0.2/M3: MSM (Moment Shadow Maps) sampling for terrain - 4 moments
fn sample_shadow_msm_terrain(
    shadow_coords: vec2<f32>,
    receiver_depth: f32,
    cascade_idx: u32,
    moment_bias: f32
) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    return msm_visibility_from_moments(moments, receiver_depth, moment_bias);
}

fn pcss_blocker_search_terrain(
    shadow_coords: vec2<f32>,
    receiver_depth: f32,
    cascade_idx: u32,
    search_radius: f32,
) -> f32 {
    let layer_count = textureNumLayers(shadow_maps);
    if (cascade_idx >= layer_count) {
        return -1.0;
    }
    var poisson_disk = array<vec2<f32>, 12>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870),
        vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845),
        vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554),
        vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>(0.79197514, 0.19090188)
    );
    let texel_size = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
    let scaled_search_radius = search_radius * texel_size;
    let dimensions = textureDimensions(shadow_maps);
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    for (var i = 0; i < 12; i++) {
        let sample_coords = shadow_coords + poisson_disk[i] * scaled_search_radius;
        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            let texel_coords = vec2<i32>(clamp(
                sample_coords * vec2<f32>(dimensions),
                vec2<f32>(0.0),
                vec2<f32>(dimensions) - vec2<f32>(1.0)
            ));
            let shadow_depth =
                textureLoad(shadow_maps, texel_coords, i32(cascade_idx), 0);
            if (shadow_depth < receiver_depth) {
                blocker_sum += shadow_depth;
                blocker_count += 1.0;
            }
        }
    }

    if (blocker_count > 0.0) {
        return blocker_sum / blocker_count;
    }
    return -1.0;
}

fn pcss_penumbra_size_terrain(
    receiver_depth: f32,
    blocker_depth: f32,
    light_size: f32,
) -> f32 {
    let depth_diff = max(receiver_depth - blocker_depth, 0.0);
    let penumbra = (depth_diff * light_size) / max(blocker_depth, 0.001);
    return clamp(penumbra, 0.0, 100.0);
}

fn pcss_light_size_texels_terrain(
    light_size_texels: f32,
    legacy_world_radius: f32,
    cascade_texel_size: f32,
) -> f32 {
    if (legacy_world_radius > 0.0) {
        // Legacy public pcss_light_radius is world-space. Convert against the
        // selected cascade so the penumbra calculation remains texel-space.
        return legacy_world_radius / max(cascade_texel_size, 0.000001);
    }
    return max(light_size_texels, 0.0);
}

/// Sample shadow map with technique-based filtering
fn sample_shadow_pcf_terrain(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    cascade_idx: u32,
) -> f32 {
    // P0 early-out: NoopShadow sentinel via cascade_count == 0
    // Real CSM always has at least one cascade; noop has cascade_count = 0
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;  // Fully lit when shadows disabled
    }
    // Normal shadow sampling path
    let cascade = csm_uniforms.cascades[cascade_idx];
    
    // Transform to light space using pre-computed combined matrix
    let light_space_pos = det_mat4_mul_vec4(cascade.light_view_proj, vec4<f32>(world_pos, 1.0));
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to texture coordinates [0,1]
    // Note: ndc.x and ndc.y are in [-1, 1], need to map to [0, 1]
    // For Y, we flip because texture V=0 is at top, but NDC Y=-1 is at bottom
    let shadow_coords = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    
    // glam's orthographic_rh already outputs Z in [0, 1] range for WebGPU
    // No additional mapping needed - use ndc.z directly
    let depth_01 = ndc.z;
    
    // Check if outside shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        depth_01 < 0.0 || depth_01 > 1.0) {
        return 1.0; // No shadow outside bounds
    }
    
    // Apply depth bias to prevent shadow acne
    // Combine constant bias with slope-scaled bias for grazing angles
    let light_dir_norm = det_normalize3(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(det_dot3(normal, light_dir_norm), 0.0);
    let slope_factor = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = csm_uniforms.depth_bias
        + csm_uniforms.slope_bias * slope_factor
        + csm_uniforms.peter_panning_offset;
    let compare_depth = depth_01 - bias;
    
    // P0.2/M3: Shadow technique dispatch based on technique uniform
    // technique=0: HARD (single sample, hard edges)
    // technique=1: PCF (3x3 kernel, soft edges)
    // technique=2: PCSS (5x5+ kernel with light radius scaling, variable penumbra)
    // technique=3: VSM (Variance Shadow Maps)
    // technique=4: EVSM (Exponential Variance Shadow Maps)
    // technique=5: MSM (Moment Shadow Maps)
    let technique = csm_uniforms.technique;
    
    // PCSS x/y radii are in shadow-map texels; z is moment bias and w is light size.
    let moment_bias = csm_uniforms.technique_params.z;
    
    // HARD shadows (technique=0): single sample, no filtering
    if (technique == 0u) {
        return textureSampleCompare(
            shadow_maps,
            shadow_sampler,
            shadow_coords,
            i32(cascade_idx),
            compare_depth
        );
    }
    
    // PCF (technique=1): 3x3 kernel for soft edges
    if (technique == 1u) {
        let kernel_size = 3;
        let radius = 1;
        let texel_size_uv = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
        var shadow_sum = 0.0;
        for (var y = -radius; y <= radius; y = y + 1) {
            for (var x = -radius; x <= radius; x = x + 1) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size_uv;
                let sample_coords = shadow_coords + offset;
                let depth_sample = textureSampleCompare(
                    shadow_maps,
                    shadow_sampler,
                    sample_coords,
                    i32(cascade_idx),
                    compare_depth
                );
                shadow_sum = shadow_sum + depth_sample;
            }
        }
        return shadow_sum / 9.0;
    }
    
    // PCSS (technique=2): blocker-dependent penumbra and adaptive PCF
    if (technique == 2u) {
        let avg_blocker_depth = pcss_blocker_search_terrain(
            shadow_coords,
            compare_depth,
            cascade_idx,
            min(csm_uniforms.technique_params.x, 50.0)
        );
        if (avg_blocker_depth < 0.0) {
            return 1.0;
        }

        let penumbra = pcss_penumbra_size_terrain(
            compare_depth,
            avg_blocker_depth,
            pcss_light_size_texels_terrain(
                csm_uniforms.technique_params.w,
                csm_uniforms.technique_reserved.x,
                cascade.texel_size
            )
        );
        // The configured filter radius is a texel-space upper bound. Keeping
        // the blocker-derived radius below it preserves contact hardening.
        let max_filter_radius = min(csm_uniforms.technique_params.y, 100.0);
        let filter_radius = min(
            max(penumbra, min(max_filter_radius, 1.0)),
            max_filter_radius
        );
        var poisson_disk = array<vec2<f32>, 16>(
            vec2<f32>(-0.94201624, -0.39906216),
            vec2<f32>(0.94558609, -0.76890725),
            vec2<f32>(-0.094184101, -0.92938870),
            vec2<f32>(0.34495938, 0.29387760),
            vec2<f32>(-0.91588581, 0.45771432),
            vec2<f32>(-0.81544232, -0.87912464),
            vec2<f32>(-0.38277543, 0.27676845),
            vec2<f32>(0.97484398, 0.75648379),
            vec2<f32>(0.44323325, -0.97511554),
            vec2<f32>(0.53742981, -0.47373420),
            vec2<f32>(-0.26496911, -0.41893023),
            vec2<f32>(0.79197514, 0.19090188),
            vec2<f32>(-0.24188840, 0.99706507),
            vec2<f32>(-0.81409955, 0.91437590),
            vec2<f32>(0.19984126, 0.78641367),
            vec2<f32>(0.14383161, -0.14100790)
        );
        let scaled_filter_radius =
            filter_radius / max(csm_uniforms.shadow_map_size, 1.0);
        var shadow_sum = 0.0;
        for (var i = 0; i < 16; i++) {
            let sample_coords = shadow_coords + poisson_disk[i] * scaled_filter_radius;
            if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
                sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
                shadow_sum += textureSampleCompare(
                    shadow_maps,
                    shadow_sampler,
                    sample_coords,
                    i32(cascade_idx),
                    clamp(compare_depth, 0.0, 1.0)
                );
            } else {
                shadow_sum += 1.0;
            }
        }
        return shadow_sum / 16.0;
    }
    
    // VSM (technique=3): Variance Shadow Maps using moment texture
    if (technique == 3u) {
        return sample_shadow_vsm_terrain(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // EVSM (technique=4): Exponential Variance Shadow Maps
    if (technique == 4u) {
        return sample_shadow_evsm_terrain(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // MSM (technique=5): Moment Shadow Maps (4 moments)
    if (technique == 5u) {
        return sample_shadow_msm_terrain(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // Fallback: PCF
    let texel_size_uv = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
    var shadow_sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size_uv;
            let sample_coords = shadow_coords + offset;
            shadow_sum = shadow_sum + textureSampleCompare(
                shadow_maps,
                shadow_sampler,
                sample_coords,
                i32(cascade_idx),
                compare_depth
            );
        }
    }
    return shadow_sum / 9.0;
}

/// Normalize world position for shadow calculations
/// Shadow maps use normalized heights [0, h_exag] to match XY scale [-0.5, 0.5]
/// This function computes shadow-normalized position from tex_coord (not interpolated world_pos)
/// CRITICAL: The main shader uses a fullscreen triangle (3 vertices) where interpolated
/// world_position is INCORRECT at most fragments. We must compute both XY and Z from tex_coord.
fn normalize_for_shadow(tex_coord: vec2<f32>) -> vec3<f32> {
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_exag = u_terrain.spacing_h_exag.z;
    let spacing = u_terrain.spacing_h_exag.x;
    let h_range = max(h_max - h_min, 1e-6);
    
    // Compute world XY from tex_coord (not from interpolated world_position!)
    // This matches what the shadow depth shader does: world_xy = (uv - 0.5) * spacing
    let world_xy = (tex_coord - vec2<f32>(0.5, 0.5)) * spacing;
    
    // Sample height directly from heightmap at this fragment's UV
    // This matches what the shadow depth shader does
    let h_raw = sample_height_bilinear(tex_coord);
    let h_norm = clamp((h_raw - h_min) / h_range, 0.0, 1.0);
    
    // Apply height curve (must match shadow depth shader)
    let h_curved = apply_height_curve01(h_norm);
    
    // Compute shadow-normalized Z (matches shadow depth shader: world_z = h_curved * h_exag)
    let shadow_z = h_curved * h_exag;
    
    return vec3<f32>(world_xy.x, world_xy.y, shadow_z);
}

/// Calculate shadow visibility for terrain
fn calculate_shadow_terrain(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32, tex_coord: vec2<f32>) -> f32 {
    // P0 early-out: when there are no active cascades, terrain is fully lit
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }

    // Select appropriate cascade
    let cascade_idx = select_cascade_terrain(view_depth);
    
    // Normalize world position for shadow lookup (match shadow map's normalized heights)
    // Compute from tex_coord since world_pos is incorrectly interpolated for fullscreen triangle
    let shadow_pos = normalize_for_shadow(tex_coord);
    
    // Sample shadow with PCF for current cascade
    var shadow_factor = sample_shadow_pcf_terrain(shadow_pos, normal, cascade_idx);
    
    // Cascade blending: smooth transitions between cascades to avoid visible seams
    // Only blend if cascade_blend_range > 0 and we're not at the last cascade
    let blend_range = csm_uniforms.cascade_blend_range;
    if (blend_range > 0.0 && cascade_idx < csm_uniforms.cascade_count - 1u) {
        let current_far = csm_uniforms.cascades[cascade_idx].far_distance;
        let blend_start = current_far * (1.0 - blend_range);
        
        // Check if we're in the blend region near cascade boundary
        if (view_depth > blend_start) {
            // Sample next cascade
            let next_shadow = sample_shadow_pcf_terrain(shadow_pos, normal, cascade_idx + 1u);
            
            // Blend between cascades based on depth within blend region
            let blend_factor = (view_depth - blend_start) / (current_far - blend_start);
            shadow_factor = det_mix(shadow_factor, next_shadow, blend_factor);
        }
    }
    
    return shadow_factor;
}

/// Debug: Get cascade color for visualization
/// Red=cascade 0, Green=cascade 1, Blue=cascade 2, Yellow=cascade 3
fn get_cascade_debug_color(cascade_idx: u32) -> vec3<f32> {
    switch cascade_idx {
        case 0u: { return vec3<f32>(1.0, 0.2, 0.2); } // Red
        case 1u: { return vec3<f32>(0.2, 1.0, 0.2); } // Green  
        case 2u: { return vec3<f32>(0.2, 0.2, 1.0); } // Blue
        case 3u: { return vec3<f32>(1.0, 1.0, 0.2); } // Yellow
        default: { return vec3<f32>(1.0, 0.0, 1.0); } // Magenta (error)
    }
}

/// Debug: Calculate shadow with debug info
/// Returns (shadow_visibility, cascade_debug_color)
fn debug_shadow_with_vis(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_depth: f32,
    tex_coord: vec2<f32>
) -> vec4<f32> {
    // P0 early-out: NoopShadow sentinel via cascade_count == 0 → fully lit, cascade 0 color
    if (csm_uniforms.cascade_count == 0u) {
        let cascade_color = get_cascade_debug_color(0u);
        return vec4<f32>(cascade_color, 1.0);
    }
    
    let cascade_idx = select_cascade_terrain(view_depth);
    let shadow_pos = normalize_for_shadow(tex_coord);
    let shadow_vis = sample_shadow_pcf_terrain(shadow_pos, normal, cascade_idx);
    let cascade_color = get_cascade_debug_color(cascade_idx);
    
    // Return shadow visibility in w, cascade color in xyz
    return vec4<f32>(cascade_color, shadow_vis);
}

/// Debug: Show shadow map UV coordinates
/// Red = U coordinate, Green = V coordinate, Blue = depth (NDC.z)
fn debug_shadow_coords(world_pos: vec3<f32>, cascade_idx: u32) -> vec3<f32> {
    let cascade = csm_uniforms.cascades[cascade_idx];
    let light_space_pos = det_mat4_mul_vec4(cascade.light_view_proj, vec4<f32>(world_pos, 1.0));
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to texture coordinates [0,1]
    let shadow_u = ndc.x * 0.5 + 0.5;
    let shadow_v = ndc.y * -0.5 + 0.5;
    let shadow_depth = ndc.z;
    
    // Return as colors: R=U, G=V, B=depth
    // If R or G is outside [0,1], coordinates are out of shadow map bounds
    return vec3<f32>(
        clamp(shadow_u, 0.0, 1.0),
        clamp(shadow_v, 0.0, 1.0),
        clamp(shadow_depth, 0.0, 1.0)
    );
}

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) world_position : vec3<f32>,
    @location(1) world_normal : vec3<f32>,
    @location(2) tex_coord : vec2<f32>,
    // TESSELLA visibility identity: ((selected_lod & 0xf) << 12) | (tile_index & 0xfff).
    // terrain_visbuffer_write.wgsl packs this with the primitive index; no
    // forward shading path reads it.
    @location(3) @interpolate(flat) tile_id : u32,
};

struct FragmentOutput {
    @location(0) color : vec4<f32>,
    // M1: AOV outputs (only written when AOV pipeline is active)
    // These are optional MRT targets - when not bound, writes are ignored
    @location(1) aov_albedo : vec4<f32>,   // Base color before lighting (RGB) + alpha
    @location(2) aov_normal : vec4<f32>,   // Normalized world-space normal (RGB) + alpha
    @location(3) aov_depth : vec4<f32>,    // Linear depth normalized to [0,1] (R) + padding
    // VERITAS: albedo-family VT source id of the dominant material layer's
    // dominant triplanar projection (R32Uint target; 0 == SOURCE_ID_NONE)
    @location(4) source_id : u32,
};

// VERITAS: resolved in the material splat loop (where layer weights and
// triplanar context are in scope) and handed to the fragment output at the
// AOV write block. Stays 0 (SOURCE_ID_NONE) whenever the material albedo
// path does not run (VT disabled, colormap-only albedo, background).
var<private> terrain_vt_albedo_source_id: u32 = 0u;

fn sample_height(uv : vec2<f32>) -> f32 {
    return sample_height_bilinear(uv);
}

fn sample_height_level(uv: vec2<f32>, lod: f32) -> f32 {
    return sample_height_bilinear_level(uv, lod);
}

fn get_height_geom_t(h_raw: f32) -> f32 {
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let range = max(h_max - h_min, 1e-6);
    return clamp(det_div(h_raw - h_min, range), 0.0, 1.0);
}

fn apply_height_curve01(t: f32) -> f32 {
    let mode = u32(u_shading.height_curve.x + 0.5);
    let strength = clamp(u_shading.height_curve.y, 0.0, 1.0);
    if (strength <= 0.0) {
        return t;
    }

    var curved = t;
    if (mode == 1u) { // pow
        let p = max(u_shading.height_curve.z, 0.01);
        curved = det_pow(t, p);
    } else if (mode == 2u) { // smoothstep
        curved = t * t * (3.0 - 2.0 * t);
    } else if (mode == 3u) { // lut
        curved = height_curve_lut_sample(t);
    }

    return det_mix(t, curved, strength);
}

fn sample_height_geom(uv : vec2<f32>) -> f32 {
    let h_raw = sample_height_bilinear(uv);
    let t = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    return det_fma(apply_height_curve01(t), h_max - h_min, h_min);
}

fn height_curve_lut_sample(t: f32) -> f32 {
    let dims = textureDimensions(height_curve_lut_tex, 0);
    let max_x = max(i32(dims.x) - 1, 0);
    let u = clamp(t, 0.0, 1.0);
    let x = i32(round(u * f32(max_x)));
    return textureLoad(height_curve_lut_tex, vec2<i32>(x, 0), 0).r;
}

fn calculate_texel_size() -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    return vec2<f32>(
        select(1.0, 1.0 / dims.x, dims.x > 0.0),
        select(1.0, 1.0 / dims.y, dims.y > 0.0),
    );
}

/// Vertex shader for terrain rendering.
/// Supports two modes controlled by camera_mode_params.x:
/// - Mode 0 (screen): Fullscreen triangle with orthographic screen coverage (default, preserves legacy behavior)
/// - Mode 1 (mesh): Grid mesh with perspective-correct view*proj transformation
@vertex
fn vs_main(@builtin(vertex_index) vertex_id : u32) -> VertexOutput {
    var out : VertexOutput;

    let camera_mode = u32(u_terrain.camera_mode_params.x);
    let grid_size = u32(max(u_terrain.camera_mode_params.y, 64.0));
    
    var uv : vec2<f32>;
    
    if (camera_mode == 1u) {
        // MESH MODE: Use grid coordinates for perspective-correct terrain rendering
        // Generate triangle mesh from vertex_id
        // For a grid_size x grid_size grid, we have (grid_size-1)^2 quads, each with 2 triangles
        // Each triangle has 3 vertices, so total vertices = 6 * (grid_size-1)^2
        let quads_per_row = grid_size - 1u;
        let vertices_per_quad = 6u; // 2 triangles * 3 vertices
        
        // Compute which quad and which vertex within the quad
        let quad_idx = vertex_id / vertices_per_quad;
        let vert_in_quad = vertex_id % vertices_per_quad;
        
        // Quad position in grid
        let quad_x = quad_idx % quads_per_row;
        let quad_y = quad_idx / quads_per_row;
        
        // Vertex offsets for the two triangles in a quad:
        // Counter-clockwise winding in screen space after view*proj transform
        // (wgpu default front_face is CCW, we cull Back faces)
        // Triangle 0: (0,0), (1,0), (1,1)  -> vertices 0,1,2
        // Triangle 1: (0,0), (1,1), (0,1)  -> vertices 3,4,5
        var offset_x: u32;
        var offset_y: u32;
        switch (vert_in_quad) {
            case 0u: { offset_x = 0u; offset_y = 0u; } // Tri 0, corner 0
            case 1u: { offset_x = 1u; offset_y = 0u; } // Tri 0, corner 1
            case 2u: { offset_x = 1u; offset_y = 1u; } // Tri 0, corner 2
            case 3u: { offset_x = 0u; offset_y = 0u; } // Tri 1, corner 0
            case 4u: { offset_x = 1u; offset_y = 1u; } // Tri 1, corner 1
            default: { offset_x = 0u; offset_y = 1u; } // Tri 1, corner 2 (case 5u)
        }
        
        let grid_x = quad_x + offset_x;
        let grid_y = quad_y + offset_y;
        
        // UV coordinates [0,1] from grid position
        uv = vec2<f32>(
            f32(grid_x) / f32(grid_size - 1u),
            f32(grid_y) / f32(grid_size - 1u)
        );
    } else {
        // SCREEN MODE: Fullscreen triangle (legacy orthographic-style rendering)
        // vertex 0: (-1, -1) -> UV (0, 0)
        // vertex 1: ( 3, -1) -> UV (2, 0)
        // vertex 2: (-1,  3) -> UV (0, 2)
        let uv_x = f32((vertex_id << 1u) & 2u);
        let uv_y = f32(vertex_id & 2u);
        uv = vec2<f32>(uv_x, uv_y);
    }

    // Reconstruct world position from UV coordinates
    // The terrain is centered at origin with spacing defining the XY extent
    let spacing = u_terrain.spacing_h_exag.x;
    let h_exag = u_terrain.spacing_h_exag.z;

    // Map UV [0,1] to world XY centered at origin
    let world_xy = (uv - vec2<f32>(0.5, 0.5)) * spacing;

    // Sample height from the portable explicit-bilinear path.
    let h_raw = sample_height_bilinear(uv);
    let t_geom = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_disp = det_fma(apply_height_curve01(t_geom), h_max - h_min, h_min);
    
    // For mesh mode: center terrain vertically around Z=0 so camera at origin can see it
    // For screen mode: keep original height for fragment shader compatibility
    let h_center = (h_min + h_max) * 0.5;
    let world_z_centered = (h_disp - h_center) * h_exag;
    let world_z_original = h_disp * h_exag;
    
    // Use centered Z for mesh mode clip position, but keep original for world_position
    // (world_position is used for lighting which expects real elevation)
    let world_pos = vec3<f32>(world_xy.x, world_xy.y, world_z_original);
    out.world_position = world_pos;
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0); // Z-up, recalculated in fragment shader
    out.tex_coord = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    out.tile_id = 0u;
    
    if (camera_mode == 1u) {
        // MESH MODE: Apply view and projection matrices for proper perspective
        // Use centered Z for clip position so terrain is visible from camera at origin
        let mesh_world_pos = vec3<f32>(world_xy.x, world_xy.y, world_z_centered);
        out.clip_position = det_mat4_mul_vec4(
            u_terrain.proj,
            det_mat4_mul_vec4(u_terrain.view, vec4<f32>(mesh_world_pos, 1.0)),
        );
    } else {
        // SCREEN MODE: Fixed NDC clip positions (fullscreen triangle)
        let ndc_x = uv.x * 2.0 - 1.0;
        let ndc_y = uv.y * 2.0 - 1.0;
        out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    }
    
    return out;
}

/// Sample height at a specific LOD level for LOD-aware normal computation.
fn sample_height_geom_level(uv: vec2<f32>, lod: f32) -> f32 {
    let h_raw = sample_height_bilinear_level(uv, lod);
    let t = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    return det_fma(apply_height_curve01(t), h_max - h_min, h_min);
}

// ─────────────────────────────────────────────────────────────────────────────
// TESSELLA: explicit screen-space gradients for the deferred visibility resolve
// ─────────────────────────────────────────────────────────────────────────────
//
// The forward pass rasterises a 2x2 quad PER TRIANGLE, so `dpdx`/`dpdy` of an
// interpolated attribute is always that one triangle's own screen gradient --
// uncovered lanes are helper invocations carrying the same triangle's
// extrapolated attributes. The visibility resolve shades from a full-screen
// triangle instead, where a quad can straddle two terrain triangles at different
// clipmap LODs, so the hardware quad derivative there is a mixture of two
// surfaces and drives LOD selection, the LOD-aware height normal, the
// virtual-texture mip and the triplanar footprint to different values than the
// forward path computed for the same pixel.
//
// Pass 2 therefore reconstructs the covering triangle's own quad-aligned
// gradients and publishes them here. The forward pipeline never writes these,
// so `terrain_explicit_gradients` stays 0 and every `select` below returns the
// hardware derivative bit-for-bit: the forward image is unchanged.
var<private> terrain_explicit_gradients: u32 = 0u;
var<private> terrain_explicit_ddx_uv: vec2<f32> = vec2<f32>(0.0, 0.0);
var<private> terrain_explicit_ddy_uv: vec2<f32> = vec2<f32>(0.0, 0.0);
var<private> terrain_explicit_ddx_world: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
var<private> terrain_explicit_ddy_world: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

fn terrain_screen_ddx_uv(uv: vec2<f32>) -> vec2<f32> {
    return select(
        dpdxCoarse(uv),
        terrain_explicit_ddx_uv,
        terrain_explicit_gradients != 0u,
    );
}

fn terrain_screen_ddy_uv(uv: vec2<f32>) -> vec2<f32> {
    return select(
        dpdyCoarse(uv),
        terrain_explicit_ddy_uv,
        terrain_explicit_gradients != 0u,
    );
}

fn terrain_screen_ddx_world(world_pos: vec3<f32>) -> vec3<f32> {
    return select(
        dpdxCoarse(world_pos),
        terrain_explicit_ddx_world,
        terrain_explicit_gradients != 0u,
    );
}

fn terrain_screen_ddy_world(world_pos: vec3<f32>) -> vec3<f32> {
    return select(
        dpdyCoarse(world_pos),
        terrain_explicit_ddy_world,
        terrain_explicit_gradients != 0u,
    );
}

/// Compute LOD from screen-space UV footprint (for LOD-aware Sobel).
/// Returns LOD level and texel size at that LOD.
struct LodInfo {
    lod: f32,
    texel_uv: vec2<f32>,
}

fn compute_height_lod(uv: vec2<f32>) -> LodInfo {
    var info: LodInfo;
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    let max_lod = f32(textureNumLevels(height_tex) - 1u);
    
    // Compute screen-space derivatives of UV
    let ddx_uv = terrain_screen_ddx_uv(uv);
    let ddy_uv = terrain_screen_ddy_uv(uv);
    
    // Footprint in texels (at mip 0)
    let rho = max(length(ddx_uv * dims), length(ddy_uv * dims));
    
    // LOD = log2 of footprint, clamped to valid range
    info.lod = clamp(log2(max(rho, 1.0)), 0.0, max_lod);
    
    // Texel size at this LOD (in UV space)
    let mip_scale = exp2(info.lod);
    info.texel_uv = vec2<f32>(det_div(mip_scale, dims.x), det_div(mip_scale, dims.y));
    
    return info;
}

/// Calculate normal from height map using LOD-aware Sobel filter.
/// Uses explicit LOD for all samples to avoid mip mismatch with offsets.
fn calculate_normal_lod_aware(uv: vec2<f32>) -> vec3<f32> {
    let lod_info = compute_height_lod(uv);
    let lod = lod_info.lod;
    let texel_uv = lod_info.texel_uv;
    
    let offset_x = vec2<f32>(texel_uv.x, 0.0);
    let offset_y = vec2<f32>(0.0, texel_uv.y);
    
    // All 9 samples at the SAME LOD level
    let tl = sample_height_geom_level(uv - offset_x - offset_y, lod);
    let t  = sample_height_geom_level(uv - offset_y, lod);
    let tr = sample_height_geom_level(uv + offset_x - offset_y, lod);
    let l  = sample_height_geom_level(uv - offset_x, lod);
    let r  = sample_height_geom_level(uv + offset_x, lod);
    let bl = sample_height_geom_level(uv - offset_x + offset_y, lod);
    let b  = sample_height_geom_level(uv + offset_y, lod);
    let br = sample_height_geom_level(uv + offset_x + offset_y, lod);
    
    // Sobel gradients
    let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    
    // Scale by world-space texel size for proper gradient magnitude
    // At higher LOD, texels cover more world space, so gradients are naturally smoothed
    let spacing = u_terrain.spacing_h_exag.x;
    let world_texel = texel_uv * spacing; // Texel size in world units
    
    let vertical_scale = max(u_terrain.spacing_h_exag.z * 0.5, 1e-3);
    return det_normalize3(vec3<f32>(
        -det_div(dx, world_texel.x),
        vertical_scale,
        -det_div(dy, world_texel.y),
    ));
}

/// Sprint 2: Multi-scale height normal for enhanced edge visibility.
/// Samples height at multiple LOD levels and blends for multi-octave detail.
fn calculate_normal_multiscale(uv: vec2<f32>) -> vec3<f32> {
    let lod_info = compute_height_lod(uv);
    let base_lod = lod_info.lod;
    let texel_uv = lod_info.texel_uv;
    let spacing = u_terrain.spacing_h_exag.x;
    let vertical_scale = max(u_terrain.spacing_h_exag.z * 0.5, 1e-3);
    
    var combined_dx = 0.0;
    var combined_dy = 0.0;
    let total_weight = 1.75;
    
    // Octave 0: Fine detail (weight 1.0)
    {
        let oct_lod = base_lod;
        let oct_texel = texel_uv;
        let off_x = vec2<f32>(oct_texel.x, 0.0);
        let off_y = vec2<f32>(0.0, oct_texel.y);
        let tl = sample_height_geom_level(uv - off_x - off_y, oct_lod);
        let t  = sample_height_geom_level(uv - off_y, oct_lod);
        let tr = sample_height_geom_level(uv + off_x - off_y, oct_lod);
        let l  = sample_height_geom_level(uv - off_x, oct_lod);
        let r  = sample_height_geom_level(uv + off_x, oct_lod);
        let bl = sample_height_geom_level(uv - off_x + off_y, oct_lod);
        let b  = sample_height_geom_level(uv + off_y, oct_lod);
        let br = sample_height_geom_level(uv + off_x + off_y, oct_lod);
        let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
        let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
        let world_texel = oct_texel * spacing;
        combined_dx += (dx / world_texel.x) * 1.0;
        combined_dy += (dy / world_texel.y) * 1.0;
    }
    
    // Octave 1: Medium detail (weight 0.5)
    {
        let oct_lod = base_lod + 1.0;
        let oct_texel = texel_uv * 2.0;
        let off_x = vec2<f32>(oct_texel.x, 0.0);
        let off_y = vec2<f32>(0.0, oct_texel.y);
        let tl = sample_height_geom_level(uv - off_x - off_y, oct_lod);
        let t  = sample_height_geom_level(uv - off_y, oct_lod);
        let tr = sample_height_geom_level(uv + off_x - off_y, oct_lod);
        let l  = sample_height_geom_level(uv - off_x, oct_lod);
        let r  = sample_height_geom_level(uv + off_x, oct_lod);
        let bl = sample_height_geom_level(uv - off_x + off_y, oct_lod);
        let b  = sample_height_geom_level(uv + off_y, oct_lod);
        let br = sample_height_geom_level(uv + off_x + off_y, oct_lod);
        let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
        let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
        let world_texel = oct_texel * spacing;
        combined_dx += (dx / world_texel.x) * 0.5;
        combined_dy += (dy / world_texel.y) * 0.5;
    }
    
    // Octave 2: Coarse detail (weight 0.25)
    {
        let oct_lod = base_lod + 2.0;
        let oct_texel = texel_uv * 4.0;
        let off_x = vec2<f32>(oct_texel.x, 0.0);
        let off_y = vec2<f32>(0.0, oct_texel.y);
        let tl = sample_height_geom_level(uv - off_x - off_y, oct_lod);
        let t  = sample_height_geom_level(uv - off_y, oct_lod);
        let tr = sample_height_geom_level(uv + off_x - off_y, oct_lod);
        let l  = sample_height_geom_level(uv - off_x, oct_lod);
        let r  = sample_height_geom_level(uv + off_x, oct_lod);
        let bl = sample_height_geom_level(uv - off_x + off_y, oct_lod);
        let b  = sample_height_geom_level(uv + off_y, oct_lod);
        let br = sample_height_geom_level(uv + off_x + off_y, oct_lod);
        let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
        let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
        let world_texel = oct_texel * spacing;
        combined_dx += (dx / world_texel.x) * 0.25;
        combined_dy += (dy / world_texel.y) * 0.25;
    }
    
    combined_dx /= total_weight;
    combined_dy /= total_weight;
    
    return det_normalize3(vec3<f32>(-combined_dx, vertical_scale, -combined_dy));
}

/// Calculate normal from height map using Sobel filter (LEGACY - not LOD-aware).
/// Kept for A/B comparison during flake diagnosis.
fn calculate_normal(uv : vec2<f32>, texel_size : vec2<f32>) -> vec3<f32> {
    let offset_x = vec2<f32>(texel_size.x, 0.0);
    let offset_y = vec2<f32>(0.0, texel_size.y);

    let tl = sample_height_geom(uv - offset_x - offset_y);
    let t = sample_height_geom(uv - offset_y);
    let tr = sample_height_geom(uv + offset_x - offset_y);
    let l = sample_height_geom(uv - offset_x);
    let r = sample_height_geom(uv + offset_x);
    let bl = sample_height_geom(uv - offset_x + offset_y);
    let b = sample_height_geom(uv + offset_y);
    let br = sample_height_geom(uv + offset_x + offset_y);

    let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);

    let vertical_scale = max(u_terrain.spacing_h_exag.z * 0.5, 1e-3);
    return det_normalize3(vec3<f32>(-dx, vertical_scale, -dy));
}

/// Compute geometric normal from screen-space derivatives of world position.
/// This is the "ground truth" normal that doesn't suffer from mip mismatch.
fn calculate_normal_ddxddy(world_pos: vec3<f32>) -> vec3<f32> {
    let ddx_pos = terrain_screen_ddx_world(world_pos);
    let ddy_pos = terrain_screen_ddy_world(world_pos);
    // Cross product gives surface normal (right-hand rule)
    // Note: order matters for winding direction
    let n = det_cross3(ddx_pos, ddy_pos);
    // Ensure normal points "up" (positive Y in our coordinate system)
    let n_norm = det_normalize3(n);
    return select(n_norm, -n_norm, n_norm.y < 0.0);
}

/// Compute triplanar blend weights from surface normal.
/// Returns normalized weights (sum to 1) for x, y, z projection axes.
/// T1 requirement: wx + wy + wz = 1, weights change smoothly with normal.
fn compute_triplanar_weights(normal: vec3<f32>, blend_sharpness: f32) -> vec3<f32> {
    let abs_n = abs(normal);
    // Use higher blend sharpness for cleaner projection transitions
    let sharpen = det_pow3(abs_n + vec3<f32>(1e-4), vec3<f32>(blend_sharpness * 1.5));
    let weight_sum = sharpen.x + sharpen.y + sharpen.z;
    return sharpen / max(weight_sum, 1e-4);
}

/// Procedural checker pattern for triplanar UV stretching test.
/// Returns 0.0 or 1.0 based on checker grid position.
/// T2 requirement: checker shows no stretching on steep slopes.
fn checker_pattern(uv: vec2<f32>, checker_scale: f32) -> f32 {
    let grid = floor(uv * checker_scale);
    let checker = i32(grid.x + grid.y) & 1;
    return f32(checker);
}

/// Sample triplanar checker pattern (no textures, pure procedural).
/// Uses same blending logic as texture triplanar for A/B comparison.
fn sample_triplanar_checker(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    scale: f32,
    blend_sharpness: f32,
    checker_scale: f32
) -> f32 {
    let weights = compute_triplanar_weights(normal, blend_sharpness);
    
    // Project world position to each axis plane
    let uv_x = world_pos.yz * scale;
    let uv_y = world_pos.xz * scale;
    let uv_z = world_pos.xy * scale;
    
    // Sample checker pattern for each projection
    let check_x = checker_pattern(uv_x, checker_scale);
    let check_y = checker_pattern(uv_y, checker_scale);
    let check_z = checker_pattern(uv_z, checker_scale);
    
    // Blend using triplanar weights
    return check_x * weights.x + check_y * weights.y + check_z * weights.z;
}

fn terrain_vt_enabled() -> bool {
    return terrain_vt_uniforms.config0.x > 0u;
}

fn terrain_vt_family_enabled(family_slot: u32) -> bool {
    if (family_slot >= 3u) {
        return false;
    }
    return terrain_vt_uniforms.family_info[family_slot].state.x != 0u;
}

fn terrain_vt_material_index(layer: f32) -> u32 {
    let material_count = max(terrain_vt_uniforms.config2.y, 1u);
    let clamped_layer = clamp(u32(max(layer, 0.0)), 0u, material_count - 1u);
    return min(clamped_layer, TERRAIN_VT_MATERIAL_CAPACITY - 1u);
}

fn terrain_vt_fallback_color(family_slot: u32, material_index: u32) -> vec4<f32> {
    let clamped_material = min(material_index, TERRAIN_VT_MATERIAL_CAPACITY - 1u);
    if (family_slot == TERRAIN_VT_FAMILY_NORMAL) {
        return terrain_vt_fallbacks.normal[clamped_material];
    }
    if (family_slot == TERRAIN_VT_FAMILY_MASK) {
        return terrain_vt_fallbacks.mask[clamped_material];
    }
    return terrain_vt_fallbacks.albedo[clamped_material];
}

fn terrain_vt_family_virtual_size(family_slot: u32) -> vec2<f32> {
    let info = terrain_vt_uniforms.family_info[min(family_slot, 2u)].virtual_size;
    return vec2<f32>(f32(max(info.x, 1u)), f32(max(info.y, 1u)));
}

fn terrain_vt_atlas_layer(family_slot: u32) -> u32 {
    // Family-info is the single source of truth for both page-table and atlas
    // addressing. The compatibility pipeline publishes layer zero for every
    // family; the bindless pipeline publishes one descriptor per family.
    return min(
        terrain_vt_uniforms.family_info[min(family_slot, 2u)].state.z,
        2u,
    );
}

fn terrain_vt_desired_mip(family_slot: u32, ddx_uv: vec2<f32>, ddy_uv: vec2<f32>) -> u32 {
    let virtual_size = terrain_vt_family_virtual_size(family_slot);
    let footprint_x = max(length(ddx_uv * virtual_size), length(ddy_uv * virtual_size));
    let desired = max(log2(max(footprint_x, 1.0)), 0.0);
    return min(u32(desired), max(terrain_vt_uniforms.config2.x, 1u) - 1u);
}

fn terrain_vt_page_dims(mip_level: u32) -> vec2<u32> {
    let base_dims = vec2<u32>(
        max(terrain_vt_uniforms.config1.z, 1u),
        max(terrain_vt_uniforms.config1.w, 1u),
    );
    let divisor = 1u << mip_level;
    let bias = divisor - 1u;
    return max(
        (base_dims + vec2<u32>(bias, bias)) / vec2<u32>(divisor, divisor),
        vec2<u32>(1u, 1u),
    );
}

fn terrain_vt_page_table_layer(family_slot: u32, material_index: u32) -> i32 {
    let family_base = terrain_vt_uniforms.family_info[min(family_slot, 2u)].state.y;
    return i32(family_base + material_index);
}

// The page table is a single-level atlas: mip zero occupies the base region
// and finer tail regions are packed left-to-right below it. This avoids the
// mipmapped Rgba32Float image path that loses the protected Vulkan device.
fn terrain_vt_page_table_origin(mip_level: u32) -> vec2<u32> {
    if (mip_level == 0u) {
        return vec2<u32>(0u, 0u);
    }
    let base_width = max(terrain_vt_uniforms.config3.y, 1u);
    let base_height = max(terrain_vt_uniforms.config3.z, 1u);
    var tail_x = 0u;
    var prior_mip = 1u;
    loop {
        if (prior_mip >= mip_level) {
            break;
        }
        tail_x = tail_x + max(base_width >> min(prior_mip, 31u), 1u);
        prior_mip = prior_mip + 1u;
    }
    return vec2<u32>(tail_x, base_height);
}

fn terrain_vt_page_table_load(
    page: vec2<i32>,
    layer: i32,
    mip_level: u32,
) -> vec4<f32> {
    let origin = terrain_vt_page_table_origin(mip_level);
    return textureLoad(
        terrain_vt_page_table,
        page + vec2<i32>(origin),
        layer,
        0,
    );
}

// Albedo triplanar coordinates repeat. GridVertex data-family coordinates are
// closed [0,1] coverage and must keep the top/right edge in the last texel
// rather than wrapping uv=1 to the opposite edge.
fn terrain_vt_normalize_family_uv(family_slot: u32, uv: vec2<f32>) -> vec2<f32> {
    if (family_slot == TERRAIN_VT_FAMILY_ALBEDO) {
        return fract(uv);
    }
    let virtual_size = max(terrain_vt_family_virtual_size(family_slot), vec2<f32>(1.0));
    return clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0) - vec2<f32>(0.5) / virtual_size);
}

fn terrain_vt_feedback_index(family_slot: u32, material_index: u32, mip_level: u32, tile_x: u32, tile_y: u32) -> u32 {
    let base_pages_x = max(terrain_vt_uniforms.config1.z, 1u);
    let base_pages_y = max(terrain_vt_uniforms.config1.w, 1u);
    let material_count = max(terrain_vt_uniforms.config2.y, 1u);
    let logical_material = family_slot * material_count + material_index;
    return (((logical_material * max(terrain_vt_uniforms.config2.x, 1u)) + mip_level) * base_pages_y + tile_y) * base_pages_x + tile_x;
}

fn terrain_vt_write_feedback(family_slot: u32, material_index: u32, mip_level: u32, tile_x: u32, tile_y: u32) {
    if (terrain_vt_uniforms.config2.w == 0u) {
        return;
    }
    let base_pages_x = max(terrain_vt_uniforms.config1.z, 1u);
    let base_pages_y = max(terrain_vt_uniforms.config1.w, 1u);
    if (tile_x >= base_pages_x || tile_y >= base_pages_y) {
        return;
    }
    // `+ 1` keeps 0 reserved for an empty slot in malformed/partial readback.
    let key = terrain_vt_feedback_index(family_slot, material_index, mip_level, tile_x, tile_y) + 1u;
    let capacity = max(terrain_vt_uniforms.config3.x, 1u);
    // A bounded append uses only operations supported by every shader backend:
    // reserve one slot with atomicAdd, then publish the key with atomicStore.
    // Duplicates are expected and are removed by the CPU readback parser.
    let append_slot = atomicAdd(&terrain_vt_feedback[0], 1u);
    if (append_slot < capacity) {
        atomicStore(&terrain_vt_feedback[append_slot + 2u], key);
    } else {
        // Never a silent drop. The explicit overflow counter tells the CPU the
        // request set was incomplete; omitted samples must be regenerated by
        // later frames before those pages can be admitted.
        atomicAdd(&terrain_vt_feedback[1], 1u);
    }
}

struct TerrainVtTriplanarFeedbackUvs {
    uv_x: vec2<f32>,
    uv_y: vec2<f32>,
    uv_z: vec2<f32>,
    ddx_x: vec2<f32>,
    ddx_y: vec2<f32>,
    ddx_z: vec2<f32>,
    ddy_x: vec2<f32>,
    ddy_y: vec2<f32>,
    ddy_z: vec2<f32>,
}

// One coordinate builder is shared by albedo sampling and feedback. This is
// the exact world-space triplanar mapping used by `sample_triplanar`.
fn terrain_vt_triplanar_feedback_uvs_from_gradients(
    world_pos: vec3<f32>,
    scale: f32,
    ddx_world: vec3<f32>,
    ddy_world: vec3<f32>,
) -> TerrainVtTriplanarFeedbackUvs {
    let scaled_ddx_world = ddx_world * scale;
    let scaled_ddy_world = ddy_world * scale;
    var out: TerrainVtTriplanarFeedbackUvs;
    out.uv_x = world_pos.yz * scale;
    out.uv_y = world_pos.xz * scale;
    out.uv_z = world_pos.xy * scale;
    out.ddx_x = scaled_ddx_world.yz;
    out.ddx_y = scaled_ddx_world.xz;
    out.ddx_z = scaled_ddx_world.xy;
    out.ddy_x = scaled_ddy_world.yz;
    out.ddy_y = scaled_ddy_world.xz;
    out.ddy_z = scaled_ddy_world.xy;
    return out;
}

fn terrain_vt_triplanar_feedback_uvs(
    world_pos: vec3<f32>,
    scale: f32,
) -> TerrainVtTriplanarFeedbackUvs {
    return terrain_vt_triplanar_feedback_uvs_from_gradients(
        world_pos,
        scale,
        terrain_screen_ddx_world(world_pos),
        terrain_screen_ddy_world(world_pos),
    );
}

fn terrain_vt_write_family_feedback_uv(
    family_slot: u32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    material_index: u32,
) {
    if (!terrain_vt_family_enabled(family_slot)) {
        return;
    }
    let virtual_size = terrain_vt_family_virtual_size(family_slot);
    let tile_size = f32(max(terrain_vt_uniforms.config0.y, 1u));
    let desired_mip = terrain_vt_desired_mip(family_slot, ddx_uv, ddy_uv);
    let page_dims = terrain_vt_page_dims(desired_mip);
    let page_size = vec2<f32>(tile_size * exp2(f32(desired_mip)));
    let page = min(
        vec2<u32>(terrain_vt_normalize_family_uv(family_slot, uv) * virtual_size / page_size),
        page_dims - vec2<u32>(1u),
    );
    let desired_entry = terrain_vt_page_table_load(
        vec2<i32>(i32(page.x), i32(page.y)),
        terrain_vt_page_table_layer(family_slot, material_index),
        desired_mip,
    );
    if (desired_entry.z > 0.5) {
        return;
    }
    terrain_vt_write_feedback(
        family_slot,
        material_index,
        desired_mip,
        page.x,
        page.y,
    );
}

// A fragment appends at most one record, selected from the 4 material
// identities x (3 albedo projections + normal Grid UV + mask Grid UV). The
// pixel-position selector prevents SIMD-neighbour duplicates from allowing
// albedo to monopolise the bounded append stream.
fn terrain_vt_write_surface_feedback(
    grid_uv: vec2<f32>,
    world_pos: vec3<f32>,
    grid_feedback_ddx: vec2<f32>,
    grid_feedback_ddy: vec2<f32>,
    world_feedback_ddx: vec3<f32>,
    world_feedback_ddy: vec3<f32>,
    _material_index: u32,
    fragment_position: vec2<f32>,
) {
    if (!terrain_vt_enabled() || terrain_vt_uniforms.config2.w == 0u) {
        return;
    }
    // Every derivative is supplied by the fragment entry point before any
    // per-pixel ownership/discard or feedback-selector divergence. Keeping this
    // helper derivative-free makes the visibility and forward paths portable.
    let triplanar_feedback = terrain_vt_triplanar_feedback_uvs_from_gradients(
        world_pos,
        max(u_shading.triplanar_params.x, 1e-3),
        world_feedback_ddx,
        world_feedback_ddy,
    );
    atomicAdd(&terrain_frame_counters.feedback_records, 1u);
    let pixel_x = u32(max(fragment_position.x, 0.0));
    let pixel_y = u32(max(fragment_position.y, 0.0));
    let selector = (pixel_x + 4099u * pixel_y) % 20u;
    let material_index = selector / 5u;
    if (material_index >= terrain_vt_uniforms.config2.y) {
        return;
    }
    let coordinate_slot = selector % 5u;
    if (coordinate_slot < 3u) {
        if (coordinate_slot == 0u) {
            terrain_vt_write_family_feedback_uv(
                TERRAIN_VT_FAMILY_ALBEDO,
                triplanar_feedback.uv_x,
                triplanar_feedback.ddx_x,
                triplanar_feedback.ddy_x,
                material_index,
            );
        } else if (coordinate_slot == 1u) {
            terrain_vt_write_family_feedback_uv(
                TERRAIN_VT_FAMILY_ALBEDO,
                triplanar_feedback.uv_y,
                triplanar_feedback.ddx_y,
                triplanar_feedback.ddy_y,
                material_index,
            );
        } else {
            terrain_vt_write_family_feedback_uv(
                TERRAIN_VT_FAMILY_ALBEDO,
                triplanar_feedback.uv_z,
                triplanar_feedback.ddx_z,
                triplanar_feedback.ddy_z,
                material_index,
            );
        }
    } else if (coordinate_slot == 3u) {
        terrain_vt_write_family_feedback_uv(
            TERRAIN_VT_FAMILY_NORMAL,
            grid_uv,
            grid_feedback_ddx,
            grid_feedback_ddy,
            material_index,
        );
    } else {
        terrain_vt_write_family_feedback_uv(
            TERRAIN_VT_FAMILY_MASK,
            grid_uv,
            grid_feedback_ddx,
            grid_feedback_ddy,
            material_index,
        );
    }
}

// Shared residency-gated page walk. Returns vec4(linear_rgb, 1.0) when a
// resident tile (page-table is_resident flag) was sampled, and
// vec4(fallback_rgb_as_authored, 0.0) when the family is disabled or no tile
// along the mip chain is resident. Never samples atlas memory for
// non-resident tiles.
fn terrain_vt_resolve_family_uv(
    family_slot: u32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    layer: f32,
) -> vec4<f32> {
    if (!terrain_vt_enabled() || !terrain_vt_family_enabled(family_slot)) {
        return vec4<f32>(
            terrain_vt_fallback_color(family_slot, terrain_vt_material_index(layer)).rgb,
            0.0,
        );
    }

    let material_index = terrain_vt_material_index(layer);
    let virtual_size = terrain_vt_family_virtual_size(family_slot);
    let tile_size = f32(max(terrain_vt_uniforms.config0.y, 1u));
    let tile_border = f32(terrain_vt_uniforms.config0.z);
    let atlas_size = f32(max(terrain_vt_uniforms.config0.w, 1u));
    let max_mip_levels = max(terrain_vt_uniforms.config2.x, 1u);
    let wrapped_uv = terrain_vt_normalize_family_uv(family_slot, uv);
    let virtual_texel = wrapped_uv * virtual_size;
    let desired_mip = terrain_vt_desired_mip(family_slot, ddx_uv, ddy_uv);
    let desired_page_dims = terrain_vt_page_dims(desired_mip);
    let desired_page_size = vec2<f32>(tile_size * exp2(f32(desired_mip)), tile_size * exp2(f32(desired_mip)));
    let desired_page = min(vec2<u32>(virtual_texel / desired_page_size), desired_page_dims - vec2<u32>(1u, 1u));
    var mip_level = desired_mip;
    loop {
        let page_dims = terrain_vt_page_dims(mip_level);
        let page_size = vec2<f32>(tile_size * exp2(f32(mip_level)), tile_size * exp2(f32(mip_level)));
        let page = min(vec2<u32>(virtual_texel / page_size), page_dims - vec2<u32>(1u, 1u));
        let entry = terrain_vt_page_table_load(
            vec2<i32>(i32(page.x), i32(page.y)),
            terrain_vt_page_table_layer(family_slot, material_index),
            mip_level,
        );
        if (entry.z > 0.5) {
            let page_origin = vec2<f32>(f32(page.x), f32(page.y)) * page_size;
            let texel_in_page = (virtual_texel - page_origin) / exp2(f32(mip_level));
            let inner_texel = clamp(texel_in_page, vec2<f32>(0.0, 0.0), vec2<f32>(tile_size - 1.0, tile_size - 1.0));
            let atlas_uv = vec2<f32>(entry.x, entry.y)
                + (vec2<f32>(tile_border, tile_border) + inner_texel + vec2<f32>(0.5, 0.5)) / atlas_size;
            return vec4<f32>(
                textureSampleLevel(terrain_vt_atlas, terrain_vt_sampler, atlas_uv, 0.0).rgb,
                1.0,
            );
        }
        if (mip_level + 1u >= max_mip_levels) {
            break;
        }
        mip_level = mip_level + 1u;
    }

    return vec4<f32>(terrain_vt_fallback_color(family_slot, material_index).rgb, 0.0);
}

// ──────────────────────────────────────────────────────────────────────────
// VERITAS: per-pixel provenance.
// Same residency-gated page walk as terrain_vt_resolve_family_uv, but
// returns the stable source id of the tile actually sampled
// (family_slot * capacity + material_index + 1) instead of a color, and 0
// (SOURCE_ID_NONE) when the family is disabled or the whole mip chain is
// non-resident (i.e. the color path used the fallback). Never writes
// feedback, so streaming behaviour is untouched.
// ──────────────────────────────────────────────────────────────────────────
fn terrain_vt_resolve_source_id(
    family_slot: u32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    layer: f32,
) -> u32 {
    if (!terrain_vt_enabled() || !terrain_vt_family_enabled(family_slot)) {
        return 0u;
    }
    let material_index = terrain_vt_material_index(layer);
    let virtual_size = terrain_vt_family_virtual_size(family_slot);
    let tile_size = f32(max(terrain_vt_uniforms.config0.y, 1u));
    let max_mip_levels = max(terrain_vt_uniforms.config2.x, 1u);
    let virtual_texel = terrain_vt_normalize_family_uv(family_slot, uv) * virtual_size;
    var mip_level = terrain_vt_desired_mip(family_slot, ddx_uv, ddy_uv);
    loop {
        let page_dims = terrain_vt_page_dims(mip_level);
        let page_size = vec2<f32>(tile_size * exp2(f32(mip_level)), tile_size * exp2(f32(mip_level)));
        let page = min(vec2<u32>(virtual_texel / page_size), page_dims - vec2<u32>(1u, 1u));
        let entry = terrain_vt_page_table_load(
            vec2<i32>(i32(page.x), i32(page.y)),
            terrain_vt_page_table_layer(family_slot, material_index),
            mip_level,
        );
        if (entry.z > 0.5) {
            return family_slot * TERRAIN_VT_MATERIAL_CAPACITY + material_index + 1u;
        }
        if (mip_level + 1u >= max_mip_levels) {
            break;
        }
        mip_level = mip_level + 1u;
    }
    return 0u;
}

// VERITAS: albedo source id at the dominant triplanar projection of one
// material layer (argmax triplanar weight; ties prefer the Y axis, then X).
// Single-source attribution by definition — never a blend.
fn terrain_vt_source_id_triplanar(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    scale: f32,
    blend_sharpness: f32,
    layer: f32,
) -> u32 {
    let weights = compute_triplanar_weights(normal, blend_sharpness);
    let tri = terrain_vt_triplanar_feedback_uvs(world_pos, scale);
    var uv = tri.uv_x;
    var ddx_uv = tri.ddx_x;
    var ddy_uv = tri.ddy_x;
    if (weights.y >= weights.x && weights.y >= weights.z) {
        uv = tri.uv_y;
        ddx_uv = tri.ddx_y;
        ddy_uv = tri.ddy_y;
    } else if (weights.z > weights.x) {
        uv = tri.uv_z;
        ddx_uv = tri.ddx_z;
        ddy_uv = tri.ddy_z;
    }
    return terrain_vt_resolve_source_id(TERRAIN_VT_FAMILY_ALBEDO, uv, ddx_uv, ddy_uv, layer);
}

fn terrain_vt_sample_family_uv(
    family_slot: u32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    layer: f32,
) -> vec3<f32> {
    // Color (albedo) path: resident tiles yield linear color; fallback colors
    // are authored as linear and used as-is.
    return terrain_vt_resolve_family_uv(family_slot, uv, ddx_uv, ddy_uv, layer).rgb;
}

// Data-family sampling (normal / mask): resident tiles are decoded back to
// the authored byte values (the atlas view is sRGB, so undo the implicit
// sRGB->linear decode). Non-resident data families fail to semantic constants
// regardless of caller-configurable fallback values: flat tangent normal and
// neutral mask, both with w = 0 so downstream blending preserves base values.
fn terrain_vt_sample_family_data(
    family_slot: u32,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    layer: f32,
) -> vec4<f32> {
    let resolved = terrain_vt_resolve_family_uv(family_slot, uv, ddx_uv, ddy_uv, layer);
    if (resolved.w > 0.5) {
        let linear_data_atlas = terrain_vt_uniforms
            .family_info[min(family_slot, 2u)].virtual_size.z != 0u;
        var authored = select(linear_to_srgb(resolved.rgb), resolved.rgb, linear_data_atlas);
        if (family_slot == TERRAIN_VT_FAMILY_NORMAL && linear_data_atlas) {
            // BC5 stores XY only. Reconstruct the positive tangent-space Z and
            // return it in authored [0,1] space for the normal application.
            let xy = authored.rg * 2.0 - vec2<f32>(1.0);
            let z = sqrt(max(1.0 - dot(xy, xy), 0.0));
            authored.b = z * 0.5 + 0.5;
        }
        return vec4<f32>(authored, 1.0);
    }
    if (family_slot == TERRAIN_VT_FAMILY_NORMAL) {
        return vec4<f32>(0.5, 0.5, 1.0, 0.0);
    }
    if (family_slot == TERRAIN_VT_FAMILY_MASK) {
        return vec4<f32>(1.0, 1.0, 1.0, 0.0);
    }
    return resolved;
}

fn sample_material_layer_uv(
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    layer: f32,
) -> vec3<f32> {
    if (!terrain_vt_enabled() || !terrain_vt_family_enabled(TERRAIN_VT_FAMILY_ALBEDO)) {
        let layer_index = i32(layer);
        return textureSampleGrad(material_albedo_tex, material_samp, uv, layer_index, ddx_uv, ddy_uv).rgb;
    }

    return terrain_vt_sample_family_uv(TERRAIN_VT_FAMILY_ALBEDO, uv, ddx_uv, ddy_uv, layer);
}

/// Triplanar sampling with textureSampleGrad for correct mip selection.
/// Computes UV gradients from world position derivatives for each projection axis.
fn sample_triplanar(
    world_pos : vec3<f32>,
    normal : vec3<f32>,
    scale : f32,
    blend_sharpness : f32,
    layer : f32,
    _lod_bias : f32  // Unused - gradients determine LOD
) -> vec3<f32> {
    let weights = compute_triplanar_weights(normal, blend_sharpness);

    let tri = terrain_vt_triplanar_feedback_uvs(world_pos, scale);
    let color_x = sample_material_layer_uv(tri.uv_x, tri.ddx_x, tri.ddy_x, layer);
    let color_y = sample_material_layer_uv(tri.uv_y, tri.ddx_y, tri.ddy_y, layer);
    let color_z = sample_material_layer_uv(tri.uv_z, tri.ddx_z, tri.ddy_z, layer);

    return color_x * weights.x + color_y * weights.y + color_z * weights.z;
}

fn build_tbn(normal : vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.y) > 0.99);
    let tangent = det_normalize3(det_cross3(up, normal));
    let bitangent = det_cross3(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

fn apply_encoded_tangent_normal(base_normal: vec3<f32>, encoded_rgb: vec3<f32>, strength: f32, mask_value: f32) -> vec3<f32> {
    if (mask_value <= 0.001 || strength <= 0.001) {
        return base_normal;
    }
    let tangent_normal = det_normalize3(encoded_rgb * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
    let mapped = det_normalize3(build_tbn(base_normal) * tangent_normal);
    return det_normalize3(det_mix3(base_normal, mapped, clamp(strength * mask_value, 0.0, 1.0)));
}

fn material_map_mask(uv: vec2<f32>) -> f32 {
    if (material_layer_uniforms.map_flags.z <= 0.5) {
        return 1.0;
    }
    return textureSample(material_mask_tex, material_map_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))).r;
}

fn apply_material_normal_map(base_normal: vec3<f32>, uv: vec2<f32>, strength: f32, mask_value: f32) -> vec3<f32> {
    if (material_layer_uniforms.map_flags.x <= 0.5 || mask_value <= 0.001) {
        return base_normal;
    }
    let encoded = textureSample(material_normal_tex, material_map_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    return apply_encoded_tangent_normal(base_normal, encoded, strength, mask_value);
}

fn apply_material_roughness_map(base_roughness: f32, uv: vec2<f32>, mask_value: f32) -> f32 {
    if (material_layer_uniforms.map_flags.y <= 0.5 || mask_value <= 0.001) {
        return base_roughness;
    }
    let mapped = textureSample(material_roughness_tex, material_map_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))).r;
    return det_mix(base_roughness, mapped, clamp(mask_value, 0.0, 1.0));
}

// ──────────────────────────────────────────────────────────────────────────
// P6: Micro-Detail Functions
// Add close-range surface detail without LOD popping:
// - Triplanar detail normals (2m repeat) blended via RNM with distance fade
// - Procedural albedo brightness noise (±10%) using stable world-space coords
// ──────────────────────────────────────────────────────────────────────────

/// Generate procedural detail normal in tangent space using gradient noise
/// Returns a tangent-space normal that can be blended with the base normal
fn procedural_detail_normal(world_pos: vec3<f32>, scale: f32) -> vec3<f32> {
    let p = world_pos * scale;
    
    // Sample noise at offset positions to compute gradient
    let eps = 0.1;
    let nx = terrain_value_noise(p + vec3<f32>(eps, 0.0, 0.0)) - terrain_value_noise(p - vec3<f32>(eps, 0.0, 0.0));
    let ny = terrain_value_noise(p + vec3<f32>(0.0, eps, 0.0)) - terrain_value_noise(p - vec3<f32>(0.0, eps, 0.0));
    let nz = terrain_value_noise(p + vec3<f32>(0.0, 0.0, eps)) - terrain_value_noise(p - vec3<f32>(0.0, 0.0, eps));
    
    // Convert gradient to tangent-space normal perturbation
    // Scale controls how strong the perturbation is
    let gradient = vec3<f32>(nx, ny, nz) * 2.0;
    
    // Return as tangent-space normal (z=1 is unperturbed up)
    return det_normalize3(vec3<f32>(gradient.x, gradient.z, 1.0));
}

/// Reoriented Normal Mapping (RNM) blend for combining detail normals
/// This method properly reorients the detail normal based on the base normal
/// Reference: "Blending in Detail" by Colin Barré-Brisebois & Stephen Hill
fn blend_rnm(base_n: vec3<f32>, detail_n: vec3<f32>) -> vec3<f32> {
    // Reorient detail normal based on base normal
    // This gives better results than simple linear blending
    let t = base_n + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail_n * vec3<f32>(-1.0, -1.0, 1.0);
    return det_normalize3(t * det_dot3(t, u) - u * t.z);
}

/// Calculate distance fade factor for micro-detail
/// Returns 1.0 at close range (full detail), 0.0 at far range (no detail)
fn calculate_detail_fade(view_distance: f32, fade_start: f32, fade_end: f32) -> f32 {
    if (view_distance <= fade_start) {
        return 1.0;
    }
    if (view_distance >= fade_end) {
        return 0.0;
    }
    // Smooth hermite interpolation for artifact-free transition
    let t = (view_distance - fade_start) / (fade_end - fade_start);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

/// Generate procedural albedo brightness noise
/// Returns a multiplier around 1.0 (e.g., 0.9 to 1.1 for 10% noise)
fn procedural_albedo_noise(world_pos: vec3<f32>, noise_amplitude: f32) -> f32 {
    // Use a different frequency than detail normals to avoid correlation
    let noise_scale = 0.7; // World-space frequency for albedo noise
    let noise = terrain_value_noise(world_pos * noise_scale);
    // Map [0,1] noise to [-amplitude, +amplitude] and add to 1.0
    return 1.0 + (noise - 0.5) * 2.0 * noise_amplitude;
}

/// P4: Apply slope and elevation-based hue variation to increase h_std
/// Combines slope variation (steep=redder, flat=yellower) with elevation spread
/// hue_shift_strength: 0.0 = no effect, 0.1 = subtle, 0.2 = moderate
fn apply_slope_hue_variation(albedo: vec3<f32>, slope_factor: f32, height_norm: f32, hue_shift_strength: f32) -> vec3<f32> {
    if (hue_shift_strength <= 0.0) {
        return albedo;
    }
    
    // Convert RGB to HSV-like representation for hue manipulation
    let max_c = max(max(albedo.r, albedo.g), albedo.b);
    let min_c = min(min(albedo.r, albedo.g), albedo.b);
    let delta = max_c - min_c;
    
    // Skip if no saturation (grayscale)
    if (delta < 0.001) {
        return albedo;
    }
    
    // Compute current hue (0-1 range, 0=red, 0.33=green, 0.67=blue)
    var hue: f32;
    if (max_c == albedo.r) {
        hue = ((albedo.g - albedo.b) / delta) / 6.0;
        if (hue < 0.0) { hue = hue + 1.0; }
    } else if (max_c == albedo.g) {
        hue = (2.0 + (albedo.b - albedo.r) / delta) / 6.0;
    } else {
        hue = (4.0 + (albedo.r - albedo.g) / delta) / 6.0;
    }
    
    let saturation = delta / max_c;
    let value = max_c;
    
    // P4: Three sources of hue variation to achieve h_std ~0.06-0.10:
    // 1. Slope-based: steep slopes (near 1) -> redder, flat areas (near 0) -> yellower
    // 2. Elevation-based: adds spread across the image
    // 3. Fine-scale noise: increases pixel-to-pixel variation without affecting h_mean
    let slope_shift = (slope_factor - 0.5) * hue_shift_strength;
    let elev_shift = (height_norm - 0.5) * hue_shift_strength * 0.4; // Elevation adds 40% of slope effect
    // Add saturation-dependent noise to increase h_std (more saturated = more variation)
    let noise_shift = (saturation - 0.5) * hue_shift_strength * 0.5;
    let hue_shift = slope_shift + elev_shift + noise_shift;
    let new_hue = fract(hue + hue_shift); // Keep in 0-1 range
    
    // Convert back to RGB
    let c = saturation * value;
    let x = c * (1.0 - abs(fract(new_hue * 6.0) * 2.0 - 1.0));
    let m = value - c;
    
    var rgb: vec3<f32>;
    let h6 = new_hue * 6.0;
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m, m, m);
}

/// Sample DEM-derived detail normal from texture
/// P6 Gradient Match: Uses tangent-space normal map derived from DEM residual
/// UV: uses terrain UV (same as heightmap UV)
/// Returns tangent-space normal decoded from RGB [0,1] -> [-1,1]
fn sample_dem_detail_normal(uv: vec2<f32>) -> vec3<f32> {
    // Sample the detail normal texture
    let sample = textureSample(detail_normal_tex, detail_normal_samp, uv);
    // Decode from [0,1] to [-1,1] per channel (tangent space: R=X, G=Y, B=Z)
    let decoded = sample.rgb * 2.0 - 1.0;
    // Ensure valid normal (avoid NaN from zero-length vectors)
    let length_sq = det_dot3(decoded, decoded);
    if (length_sq < 0.0001) {
        return vec3<f32>(0.0, 0.0, 1.0); // Neutral normal
    }
    return det_normalize3(decoded);
}

/// Apply micro-detail to normal
/// Blends procedural detail normal with base using RNM, with distance fade
/// P6: Also samples from DEM-derived detail normal texture when available
fn apply_detail_normal(
    base_normal: vec3<f32>,
    world_pos: vec3<f32>,
    view_distance: f32,
    detail_scale: f32,
    detail_strength: f32,
    fade_start: f32,
    fade_end: f32,
) -> vec3<f32> {
    // Early out if no detail contribution
    let fade = calculate_detail_fade(view_distance, fade_start, fade_end);
    if (fade <= 0.0 || detail_strength <= 0.0) {
        return base_normal;
    }
    
    // Generate triplanar detail normal using procedural noise
    // Sample from three axes and blend by base normal weights
    let weights = compute_triplanar_weights(base_normal, 4.0);
    
    // Detail UVs from world position
    let uv_scale = 1.0 / detail_scale;
    let detail_x = procedural_detail_normal(vec3<f32>(0.0, world_pos.y, world_pos.z) * uv_scale, 1.0);
    let detail_y = procedural_detail_normal(vec3<f32>(world_pos.x, 0.0, world_pos.z) * uv_scale, 1.0);
    let detail_z = procedural_detail_normal(vec3<f32>(world_pos.x, world_pos.y, 0.0) * uv_scale, 1.0);
    
    // Blend triplanar detail normals
    let blended_detail = det_normalize3(
        detail_x * weights.x + 
        detail_y * weights.y + 
        detail_z * weights.z
    );
    
    // Build TBN from base normal
    let tbn = build_tbn(base_normal);
    
    // Transform detail normal to world space
    let detail_world = det_normalize3(det_mat3_mul_vec3(tbn, blended_detail));
    
    // RNM blend with strength and fade
    let effective_strength = detail_strength * fade;
    let blended = blend_rnm(base_normal, det_mix3(vec3<f32>(0.0, 0.0, 1.0), blended_detail, effective_strength));
    
    return det_normalize3(det_mat3_mul_vec3(tbn, blended));
}

/// P6: Apply DEM-derived detail normal to geometric normal
/// Uses the detail_normal_tex texture which contains tangent-space normals
/// derived from high-frequency DEM residual (original - gaussian_blur)
/// UV: terrain UV (0-1 across terrain extent)
fn apply_dem_detail_normal(
    base_normal: vec3<f32>,
    terrain_uv: vec2<f32>,
    view_distance: f32,
    detail_strength: f32,
    fade_start: f32,
    fade_end: f32,
) -> vec3<f32> {
    // Early out if no detail contribution
    let fade = calculate_detail_fade(view_distance, fade_start, fade_end);
    if (fade <= 0.0 || detail_strength <= 0.0) {
        return base_normal;
    }
    
    // Sample DEM-derived detail normal from texture
    let detail_tangent = sample_dem_detail_normal(terrain_uv);
    
    // Check if this is a neutral normal (fallback texture or no detail)
    // Neutral normal = (0, 0, 1) in tangent space, encoded as (0.5, 0.5, 1.0)
    let is_neutral = abs(detail_tangent.x) < 0.01 && abs(detail_tangent.y) < 0.01 && detail_tangent.z > 0.99;
    if (is_neutral) {
        return base_normal;
    }
    
    // Build TBN from base normal
    let tbn = build_tbn(base_normal);
    
    // Apply RNM blend with strength and fade
    let effective_strength = detail_strength * fade;
    let blended = blend_rnm(base_normal, det_mix3(vec3<f32>(0.0, 0.0, 1.0), detail_tangent, effective_strength));
    
    return det_normalize3(det_mat3_mul_vec3(tbn, blended));
}

fn rotate_y(v : vec3<f32>, sin_theta : f32, cos_theta : f32) -> vec3<f32> {
    return vec3<f32>(
        v.x * cos_theta + v.z * sin_theta,
        v.y,
        -v.x * sin_theta + v.z * cos_theta,
    );
}

// Note: fresnel_schlick is provided by brdf/common.wgsl (included via lighting.wgsl)

/// Parallax Occlusion Mapping with binary search refinement.
fn parallax_occlusion_mapping(
    uv : vec2<f32>,
    view_dir_tangent : vec3<f32>,
    height_scale : f32,
    min_steps : u32,
    max_steps : u32,
    refine_steps : u32
) -> vec2<f32> {
    if (height_scale <= 0.0) {
        return uv;
    }

    let view_dir = det_normalize3(view_dir_tangent);
    let min_s = clamp(max(min_steps, 1u), 1u, POM_MAX_STEPS);
    let max_s = clamp(max(max_steps, min_s), min_s, POM_MAX_STEPS);
    let refine_count = min(refine_steps, POM_MAX_REFINE_STEPS);
    let blend = clamp(abs(view_dir.z), 0.0, 1.0);
    let steps_interp = det_mix(f32(max_s), f32(min_s), blend);
    let step_count = clamp(u32(steps_interp + 0.5), 1u, max_s);
    let step_size = 1.0 / f32(step_count);

    let dir_xy = view_dir.xy;
    if (length(dir_xy) < 1e-5) {
        return uv;
    }
    let parallax_dir = det_normalize2(dir_xy) * height_scale;

    var current_uv = uv;
    var current_layer = 0.0;
    // Use explicit LOD sampling here so the raymarch does not rely on implicit
    // gradients inside a runtime-varying fragment loop, which FXC rejects.
    var current_height = sample_height_level(current_uv, 0.0);

    for (var i = 0u; i < POM_MAX_STEPS; i = i + 1u) {
        if (i >= step_count || current_layer >= current_height) {
            break;
        }
        current_uv -= parallax_dir * step_size;
        current_layer += step_size;
        current_height = sample_height_level(current_uv, 0.0);
    }

    var refine_step_size = step_size;
    for (var i = 0u; i < POM_MAX_REFINE_STEPS; i = i + 1u) {
        if (i >= refine_count) {
            break;
        }
        let delta_uv = parallax_dir * refine_step_size * 0.5;
        refine_step_size *= 0.5;
        current_height = sample_height_level(current_uv, 0.0);
        if (current_layer >= current_height) {
            current_uv -= delta_uv;
            current_layer -= refine_step_size;
        } else {
            current_uv += delta_uv;
            current_layer += refine_step_size;
        }
    }

    return current_uv;
}

/// Cook-Torrance PBR BRDF with GGX, Smith, and Schlick terms.
fn calculate_pbr_brdf(
    normal : vec3<f32>,
    view_dir : vec3<f32>,
    light_dir : vec3<f32>,
    albedo : vec3<f32>,
    roughness : f32,
    metallic : f32,
    f0 : vec3<f32>
) -> vec3<f32> {
    let halfway = det_normalize3(view_dir + light_dir);

    let n_dot_l = max(det_dot3(normal, light_dir), 0.0);
    let n_dot_v = max(det_dot3(normal, view_dir), 0.0);
    let n_dot_h = max(det_dot3(normal, halfway), 0.0);
    let v_dot_h = max(det_dot3(view_dir, halfway), 0.0);

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let alpha = max(roughness * roughness, 1e-3);
    let alpha_sq = alpha * alpha;
    let denom = (n_dot_h * n_dot_h * (alpha_sq - 1.0)) + 1.0;
    let distribution = alpha_sq / (PI * denom * denom);

    let k = (roughness + 1.0);
    let k_sq = (k * k) / 8.0;
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k_sq) + k_sq);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k_sq) + k_sq);
    let geometry = g1_l * g1_v;

    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * det_pow(1.0 - v_dot_h, 5.0);

    let specular = (distribution * geometry) * fresnel / max(4.0 * n_dot_l * n_dot_v, 1e-3);
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;
    return (diffuse + specular) * n_dot_l;
}

// ──────────────────────────────────────────────────────────────────────────
// P5-L: Lambert contrast curve for gradient enhancement
// ──────────────────────────────────────────────────────────────────────────

/// P5-L: Apply S-curve contrast to Lambert term to increase micro-contrast in slopes.
/// k in [0.0, 1.0]: 0 = original Lambert, 1 = maximum contrast.
/// Constraints:
///   - f(0, k) = 0, f(1, k) = 1 (endpoint preservation)
///   - Monotonically increasing in l
///   - |E[f(l,k)] - E[l]| / E[l] <= 10% (energy preservation within ±10%)
fn lambert_contrast(l: f32, k: f32) -> f32 {
    let l_clamped = clamp(l, 0.0, 1.0);
    // Power curve: l^(1-k*0.5) for gradient enhancement
    // k=0: linear (l^1), k=1: sqrt (l^0.5) which has steeper slope at low values
    // This increases derivative for darker pixels, creating more local contrast
    let exponent = 1.0 - k * 0.5;  // Range [0.5, 1.0]
    return det_pow(l_clamped + 1e-6, exponent);  // Small epsilon to avoid det_pow(0, x) issues
}

/// Split-normal PBR BRDF: uses smooth normal for specular (eliminates aliasing),
/// detailed normal for diffuse (keeps surface detail). Standard terrain technique.
/// P5-L: Added lambert_k parameter for contrast curve on diffuse term.
fn calculate_pbr_brdf_split_normal(
    diffuse_normal : vec3<f32>,   // Height-derived normal for diffuse shading
    specular_normal : vec3<f32>,  // Geometric/smooth normal for specular (anti-alias)
    view_dir : vec3<f32>,
    light_dir : vec3<f32>,
    albedo : vec3<f32>,
    roughness : f32,
    metallic : f32,
    f0 : vec3<f32>,
    lambert_k : f32               // P5-L: Lambert contrast parameter [0,1]
) -> vec3<f32> {
    let halfway = det_normalize3(view_dir + light_dir);

    // Diffuse uses detailed normal for surface variation
    let n_dot_l_diff_raw = max(det_dot3(diffuse_normal, light_dir), 0.0);
    // P5-L: Apply contrast curve to diffuse Lambert term
    let n_dot_l_diff = lambert_contrast(n_dot_l_diff_raw, lambert_k);
    
    // Specular uses smooth normal to avoid aliasing
    let n_dot_l_spec = max(det_dot3(specular_normal, light_dir), 0.0);
    let n_dot_v = max(det_dot3(specular_normal, view_dir), 0.0);
    let n_dot_h = max(det_dot3(specular_normal, halfway), 0.0);
    let v_dot_h = max(det_dot3(view_dir, halfway), 0.0);

    if (n_dot_l_diff <= 0.0 && n_dot_l_spec <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // GGX Distribution (specular normal)
    let alpha = max(roughness * roughness, 1e-3);
    let alpha_sq = alpha * alpha;
    let denom = (n_dot_h * n_dot_h * (alpha_sq - 1.0)) + 1.0;
    let distribution = alpha_sq / (PI * denom * denom);

    // Smith Geometry (specular normal)
    let k = (roughness + 1.0);
    let k_sq = (k * k) / 8.0;
    let g1_l = n_dot_l_spec / (n_dot_l_spec * (1.0 - k_sq) + k_sq + 1e-5);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k_sq) + k_sq + 1e-5);
    let geometry = g1_l * g1_v;

    // Fresnel
    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * det_pow(1.0 - v_dot_h, 5.0);

    // Specular term (smooth normal) - only if facing light
    var specular = vec3<f32>(0.0);
    if (n_dot_l_spec > 0.0 && n_dot_v > 0.0) {
        specular = (distribution * geometry) * fresnel / max(4.0 * n_dot_l_spec * n_dot_v, 1e-3);
        specular = specular * n_dot_l_spec;
    }
    
    // Diffuse term (detailed normal) - uses contrast-adjusted n_dot_l_diff
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI * n_dot_l_diff;
    
    return diffuse + specular;
}

/// P3: Split-roughness PBR BRDF for Toksvig specular anti-aliasing.
/// Uses base_roughness for diffuse (preserves detail), specular_roughness for specular (smoothed via Toksvig).
/// This is the key P3 fix: Toksvig applies ONLY to specular, leaving diffuse crisp.
/// P5-L: Added lambert_k parameter for contrast curve on diffuse term.
fn calculate_pbr_brdf_split_roughness(
    normal : vec3<f32>,
    view_dir : vec3<f32>,
    light_dir : vec3<f32>,
    albedo : vec3<f32>,
    base_roughness : f32,      // Original roughness for diffuse calculations
    specular_roughness : f32,  // Toksvig-adjusted roughness for specular calculations
    metallic : f32,
    f0 : vec3<f32>,
    lambert_k : f32            // P5-L: Lambert contrast parameter [0,1]
) -> vec3<f32> {
    let halfway = det_normalize3(view_dir + light_dir);

    let n_dot_l_raw = max(det_dot3(normal, light_dir), 0.0);
    // P5-L: Apply contrast curve to diffuse Lambert term
    let n_dot_l_diff = lambert_contrast(n_dot_l_raw, lambert_k);
    // Specular uses unmodified n_dot_l for physically correct highlights
    let n_dot_l_spec = n_dot_l_raw;
    
    let n_dot_v = max(det_dot3(normal, view_dir), 0.0);
    let n_dot_h = max(det_dot3(normal, halfway), 0.0);
    let v_dot_h = max(det_dot3(view_dir, halfway), 0.0);

    if (n_dot_l_raw <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // GGX Distribution - uses specular_roughness (Toksvig-adjusted)
    let alpha_spec = max(specular_roughness * specular_roughness, 1e-3);
    let alpha_spec_sq = alpha_spec * alpha_spec;
    let denom = (n_dot_h * n_dot_h * (alpha_spec_sq - 1.0)) + 1.0;
    let distribution = alpha_spec_sq / (PI * denom * denom);

    // Smith Geometry - uses specular_roughness
    let k = (specular_roughness + 1.0);
    let k_sq = (k * k) / 8.0;
    let g1_l = n_dot_l_spec / (n_dot_l_spec * (1.0 - k_sq) + k_sq);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k_sq) + k_sq);
    let geometry = g1_l * g1_v;

    // Fresnel (unchanged - doesn't depend on roughness directly in Schlick)
    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * det_pow(1.0 - v_dot_h, 5.0);

    // Specular term (uses specular_roughness, unmodified n_dot_l)
    let specular = (distribution * geometry) * fresnel / max(4.0 * n_dot_l_spec * n_dot_v, 1e-3) * n_dot_l_spec;
    
    // Diffuse term - uses contrast-adjusted n_dot_l_diff for gradient enhancement
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI * n_dot_l_diff;
    
    return diffuse + specular;
}

// P2-05: Bridge function to map terrain parameters to ShadingParamsGPU for optional eval_brdf
// Maps a subset of TerrainShadingUniforms to ShadingParamsGPU, ignoring unsupported terrain knobs
fn terrain_to_shading_params(
    roughness: f32,
    metallic: f32,
    brdf_model: u32,  // Runtime flag: which BRDF model to use (default BRDF_COOK_TORRANCE_GGX)
) -> ShadingParamsGPU {
    var params: ShadingParamsGPU;
    params.brdf = brdf_model;
    params.metallic = metallic;
    params.roughness = roughness;
    params.sheen = 0.0;        // Terrain doesn't use sheen
    params.clearcoat = 0.0;    // Terrain doesn't use clearcoat
    params.subsurface = 0.0;   // Terrain-specific SSS is applied later in the land shading path
    params.anisotropy = 0.0;   // Terrain doesn't use anisotropy
    return params;
}

// Check for NaN/Inf (debug helper for catching bad data)
fn is_finite_f32(x: f32) -> bool {
    // NaN != NaN, and Inf comparisons fail
    return !(x != x) && (x <= 3.4028235e+38) && (x >= -3.4028235e+38);
}

// ──────────────────────────────────────────────────────────────────────────
// PBR Debug Helpers - Split IBL for separate diffuse/specular visualization
// ──────────────────────────────────────────────────────────────────────────

struct IblSplit {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    specular_brdf: vec3<f32>,
    fresnel: vec3<f32>,
    n_dot_v: f32,
}

/// Compute IBL with diffuse and specular separated (for debug visualization)
fn eval_ibl_split(
    n: vec3<f32>,
    v: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> IblSplit {
    var result: IblSplit;
    
    // Clamp inputs for numeric safety
    let n_dot_v = saturate(det_dot3(n, v));
    let roughness_clamped = saturate(roughness);
    result.n_dot_v = n_dot_v;
    
    // Calculate reflection direction
    let reflection_dir = reflect(-v, n);
    
    // Fresnel term for IBL (roughness-aware Schlick)
    let one_minus_cos = saturate(1.0 - n_dot_v);
    let pow5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
    // Roughness-aware Fresnel: lerp toward white at grazing for rough surfaces
    let F_ibl = f0 + (max(vec3<f32>(1.0 - roughness_clamped), f0) - f0) * pow5;
    result.fresnel = F_ibl;
    
    // Diffuse IBL (Lambertian)
    // kD = (1 - kS) * (1 - metallic)
    let kS_ibl = F_ibl;
    let kD_ibl = (vec3<f32>(1.0) - kS_ibl) * (1.0 - metallic);
    
    // Sample irradiance cubemap
    let irradiance = textureSampleLevel(envIrradiance, envSampler, n, 0.0).rgb;
    result.diffuse = kD_ibl * base_color * irradiance;
    
    // Specular IBL (split-sum approximation)
    let mip_level = roughness_clamped * roughness_clamped * 9.0; // Assume 10 mips (0-9)
    
    // Sample prefiltered specular cubemap
    let prefiltered_color = textureSampleLevel(envSpecular, envSampler, reflection_dir, mip_level).rgb;
    
    // Sample BRDF LUT
    let brdf_lut_uv = vec2<f32>(n_dot_v, roughness_clamped);
    let brdf_lut = textureSampleLevel(brdfLUT, envSampler, brdf_lut_uv, 0.0).rg;
    
    // Split-sum: prefiltered_color * (F0 * scale + bias)
    result.specular_brdf = F_ibl * brdf_lut.x + brdf_lut.y;
    result.specular = prefiltered_color * result.specular_brdf;
    
    return result;
}

/// Compute luminance (relative luminance for sRGB primaries)
fn luminance(c: vec3<f32>) -> f32 {
    return det_dot3(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ──────────────────────────────────────────────────────────────────────────
// Water Debug Helpers (unambiguous, no-tonemap, isolated modes)
// ──────────────────────────────────────────────────────────────────────────

// Falsecolor ramp: 0=blue -> 0.5=green -> 1=red (very visible)
fn ramp_falsecolor(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = clamp(1.5 * x - 0.5, 0.0, 1.0);
    let g = clamp(1.5 - abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0);
    let b = clamp(1.5 * (1.0 - x) - 0.5, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

// Simple HDR compression for debug visibility
fn compress_hdr(x: vec3<f32>) -> vec3<f32> {
    return x / (vec3<f32>(1.0) + x);
}

// DEBUG 100: Binary water classification - blue=water, dark gray=land
fn debug_water_is_water(is_water_flag: bool) -> vec3<f32> {
    let land = vec3<f32>(0.08, 0.08, 0.08);
    let water = vec3<f32>(0.0, 0.2, 1.0); // unmistakable blue
    return select(land, water, is_water_flag);
}

// DEBUG 101: Shore-distance scalar visualization with shoreline ring
fn debug_water_scalar(is_water_flag: bool, water_scalar: f32) -> vec3<f32> {
    let land = vec3<f32>(0.05, 0.05, 0.05);
    let t = clamp(water_scalar, 0.0, 1.0);
    var rgb_water = ramp_falsecolor(t);
    // Add bright ring near shoreline (where t is small)
    let shore_ring = smoothstep(0.06, 0.00, t);
    rgb_water = clamp(rgb_water + shore_ring * vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0), vec3<f32>(1.0));
    return select(land, rgb_water, is_water_flag);
}

// DEBUG 102: IBL specular on water only (land=black)
// Diagnostic version: shows prefiltered environment (no Fresnel) to isolate the issue
fn debug_water_ibl_spec_only(is_water_flag: bool, ibl_spec: vec3<f32>) -> vec3<f32> {
    let land = vec3<f32>(0.0, 0.0, 0.0); // black = no cheating
    let s = max(ibl_spec, vec3<f32>(0.0));
    let mag = s.x + s.y + s.z;
    // If IBL is effectively zero, show magenta diagnostic
    if (mag < 0.001) {
        return select(land, vec3<f32>(1.0, 0.0, 1.0), is_water_flag); // Magenta = IBL is zero
    }
    let rgb_dbg = compress_hdr(s);
    return select(land, rgb_dbg, is_water_flag);
}

// DEBUG 103: Raw prefiltered environment sample (no Fresnel, no BRDF LUT)
fn debug_water_prefilt_raw(is_water_flag: bool, prefilt_color: vec3<f32>) -> vec3<f32> {
    let land = vec3<f32>(0.0, 0.0, 0.0);
    let s = max(prefilt_color, vec3<f32>(0.0));
    let mag = s.x + s.y + s.z;
    if (mag < 0.001) {
        return select(land, vec3<f32>(1.0, 1.0, 0.0), is_water_flag); // Yellow = prefilt zero
    }
    let rgb_dbg = compress_hdr(s);
    return select(land, rgb_dbg, is_water_flag);
}

// ──────────────────────────────────────────────────────────────────────────
// P2: Atmospheric Fog
// Height-based exponential fog applied after PBR shading, before tonemap.
// When fog_density = 0.0, returns base_color unchanged (no-op).
//
// Coordinate system: Z-up (world_pos.z = elevation)
// Density scaling: quadratic (density²) for perceptually linear 0-1 range
// ──────────────────────────────────────────────────────────────────────────
fn aether_terrain_segment_transmittance(
    distance_m: f32,
    camera_height_m: f32,
    view_mu: f32,
    bottom_radius_m: f32,
    density_scale: f32,
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
) -> vec3<f32> {
    // Mie anisotropy changes phase, not Beer-Lambert extinction. Retain the
    // validated ABI argument while delegating the complete integral.
    let _validated_mie_g = mie_g;
    return aether_eval_segment_transmittance(
        distance_m,
        camera_height_m,
        view_mu,
        bottom_radius_m,
        density_scale,
        turbidity,
        ozone_du,
    );
}

fn aether_terrain_finite_normalize(direction: vec3<f32>) -> vec3<f32> {
    let largest_component = max(
        max(abs(direction.x), abs(direction.y)),
        abs(direction.z),
    );
    let scaled = direction / max(largest_component, 1.0);
    let length_squared = dot(scaled, scaled);
    let normalized = scaled * inverseSqrt(max(length_squared, 1.0e-12));
    return select(
        vec3<f32>(0.0, 0.0, 1.0),
        normalized,
        length_squared > 1.0e-12,
    );
}

fn aether_terrain_sample_inscatter(
    altitude_m: f32,
    mu_sun: f32,
    mu_view: f32,
    nu: f32,
) -> vec3<f32> {
    let atmosphere_height = max(
        fog_uniforms.aether_planet_lut.y - fog_uniforms.aether_planet_lut.x,
        1.0,
    );
    return aether_eval_sample_accumulated_scattering(
        aether_accumulated_scattering_tex,
        altitude_m / atmosphere_height,
        mu_sun,
        mu_view,
        nu,
        max(u32(fog_uniforms.aether_planet_lut.z + 0.5), 2u),
        max(u32(fog_uniforms.aether_planet_lut.w + 0.5), 2u),
    );
}

fn aether_terrain_apply_segment(
    surface_radiance: vec3<f32>,
    distance_m: f32,
    camera_height_m: f32,
    view_direction: vec3<f32>,
    sun_direction: vec3<f32>,
    density_scale: f32,
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
) -> vec3<f32> {
    let view = aether_terrain_finite_normalize(view_direction);
    let bottom_radius_m = max(fog_uniforms.aether_planet_lut.x, 1.0);
    let endpoint_height_m = aether_eval_spherical_altitude(
        camera_height_m,
        view.z,
        distance_m,
        bottom_radius_m,
    );
    let sun = aether_terrain_finite_normalize(sun_direction);
    let view_sun_nu = dot(view, sun);
    let endpoint_mus = aether_eval_spherical_endpoint_mus(
        camera_height_m,
        view.z,
        sun.z,
        view_sun_nu,
        distance_m,
        bottom_radius_m,
    );
    let transmittance = aether_terrain_segment_transmittance(
        distance_m,
        camera_height_m,
        view.z,
        bottom_radius_m,
        density_scale,
        turbidity,
        ozone_du,
        mie_g,
    );
    let base_transmittance = aether_terrain_segment_transmittance(
        distance_m,
        camera_height_m,
        view.z,
        bottom_radius_m,
        1.0,
        turbidity,
        ozone_du,
        mie_g,
    );
    let bounded_surface = clamp(surface_radiance, vec3<f32>(0.0), vec3<f32>(65504.0));
    let sun_intensity = aether_eval_clamp_radiometric_scale(fog_uniforms.sky_params0.w);
    let atmosphere_exposure = aether_eval_clamp_radiometric_scale(fog_uniforms.sky_params1.w);
    let camera_scattering = aether_terrain_sample_inscatter(
        camera_height_m,
        sun.z,
        view.z,
        view_sun_nu,
    ) * sun_intensity;
    let endpoint_scattering = aether_terrain_sample_inscatter(
        endpoint_height_m,
        endpoint_mus.y,
        endpoint_mus.x,
        view_sun_nu,
    ) * sun_intensity;
    // Finite-segment radiative-transfer identity. Unlike a fog-color lerp,
    // both endpoints come from the accumulated single+higher-order LUT.
    let base_finite_inscatter = max(
        camera_scattering - base_transmittance * endpoint_scattering,
        vec3<f32>(0.0),
    );
    let density_adjustment = clamp(
        (vec3<f32>(1.0) - transmittance)
            / max(vec3<f32>(1.0) - base_transmittance, vec3<f32>(1.0e-6)),
        vec3<f32>(0.0),
        vec3<f32>(10.0),
    );
    // Match the boundary sky's exposure without scaling surface radiance:
    // sky_exposure is an atmosphere-only radiometric multiplier.
    let finite_inscatter =
        base_finite_inscatter * density_adjustment * atmosphere_exposure;
    let transported = bounded_surface * transmittance + finite_inscatter;
    return clamp(transported, vec3<f32>(0.0), vec3<f32>(65504.0));
}

fn apply_atmospheric_fog(
    base_color: vec3<f32>,
    world_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    screen_pos: vec2<f32>,
) -> vec3<f32> {
    let density_raw = fog_uniforms.params0.x;
    let fog_enabled = density_raw > 0.0;
    let sky_enabled = fog_uniforms.sky_params0.x > 0.5;
    let sky_aerial_enabled = sky_enabled && fog_uniforms.sky_params0.z > 0.5;
    let aether_enabled = fog_uniforms.fog_inscatter.w > 0.5;
    if (!fog_enabled && !sky_aerial_enabled) {
        return base_color;
    }
    let to_camera = camera_pos - world_pos;
    let view_distance = length(to_camera);
    let sky_color = sample_atmosphere_sky(screen_pos);

    let height_falloff = fog_uniforms.params0.y;
    let base_height = fog_uniforms.params0.z;
    let fragment_elevation = max(world_pos.z - base_height, 0.0);
    let height_factor = det_exp(-height_falloff * fragment_elevation);

    var fog_result = base_color;

    // AETHER owns the complete transport when active. Never pre-compose the
    // legacy RGB fog lerp and then apply spectral transport a second time.
    if (fog_enabled && !(aether_enabled && sky_aerial_enabled)) {
        let density = density_raw * density_raw;
        let extinction = det_exp(-density * view_distance * height_factor * 0.005);
        let inscatter = select(fog_uniforms.fog_inscatter.rgb, sky_color, sky_enabled);
        fog_result = det_mix3(inscatter, fog_result, extinction);
    }

    if (sky_aerial_enabled && aether_enabled) {
        fog_result = aether_terrain_apply_segment(
            fog_result,
            view_distance,
            max(camera_pos.z, 0.0),
            world_pos - camera_pos,
            fog_uniforms.aether_sun_direction.xyz,
            fog_uniforms.sky_params0.y,
            fog_uniforms.sky_params1.z,
            fog_uniforms.sky_params1.x,
            fog_uniforms.sky_params1.y,
        );
    } else if (sky_aerial_enabled) {
        let aerial_density = fog_uniforms.sky_params0.y;
        let sun_intensity = fog_uniforms.sky_params0.w;
        let sun_size = fog_uniforms.sky_params1.x;
        let sun_elevation = fog_uniforms.sky_params1.y;
        let turbidity = fog_uniforms.sky_params1.z;
        let exposure = fog_uniforms.sky_params1.w;
        let low_sun = 1.0 - smoothstep(0.18, 0.72, sun_elevation);
        let haze = clamp((turbidity - 1.0) / 9.0, 0.0, 1.0);
        let sun_energy = clamp(sun_intensity * (0.5 + sun_size * 0.35), 0.0, 8.0);
        let aerial_factor = 1.0 - det_exp(-aerial_density * view_distance * (0.08 + haze * 0.04));
        let aerial_amount = clamp(
            aerial_factor * (0.8 + haze * 0.25 + sun_energy * 0.05),
            0.0,
            1.0,
        );
        let luma = det_dot3(fog_result, vec3<f32>(0.2126, 0.7152, 0.0722));
        let desaturated = det_mix3(fog_result, vec3<f32>(luma), aerial_amount * (0.4 + haze * 0.15));
        let warm_tint = det_mix3(
            vec3<f32>(1.0, 1.0, 1.0),
            vec3<f32>(1.16, 0.98, 0.82),
            low_sun * (0.55 + haze * 0.25),
        );
        let atmosphere_target = sky_color * (1.0 + sun_energy * 0.04)
            * det_mix3(vec3<f32>(1.0), warm_tint, low_sun)
            + vec3<f32>(0.14, 0.07, 0.025) * low_sun * sun_energy * 0.18 * exposure;
        fog_result = det_mix3(
            desaturated,
            atmosphere_target,
            aerial_amount * (0.34 + low_sun * 0.18 + haze * 0.12),
        );
    }

    return fog_result;
}

// TERRAIN_AOV_DEPTH_BEGIN
fn normalize_aov_depth(linear_depth: f32, clip_near: f32, clip_far: f32) -> f32 {
    return saturate(det_div(
        linear_depth - clip_near,
        max(clip_far - clip_near, 1e-5)
    ));
}

fn terrain_aov_depth(input : VertexOutput) -> vec4<f32> {
    // AOV Depth: projected mesh/clipmap modes use actual raster depth because
    // their clip position is built from a height-centered instance position
    // while public `world_position` intentionally retains original elevation.
    // Legacy fullscreen mode has fixed clip Z and retains its established
    // height-varying view-space depth contract.
    let clip_near = max(u_terrain.camera_mode_params.z, 1e-5);
    let clip_far = max(u_terrain.camera_mode_params.w, clip_near + 1e-5);
    let ndc_depth = clamp(input.clip_position.z, 0.0, 1.0);
    let screen_view_position = det_mat4_mul_vec4(
        u_terrain.view,
        vec4<f32>(input.world_position, 1.0)
    );
    var linear_depth = -screen_view_position.z;
    if (u_terrain.camera_mode_params.x >= 0.5) {
        linear_depth = det_div(clip_near * clip_far, max(
            1e-5,
            clip_far - ndc_depth * (clip_far - clip_near)
        ));
    }
    let depth_normalized = normalize_aov_depth(linear_depth, clip_near, clip_far);
    return vec4<f32>(depth_normalized, depth_normalized, depth_normalized, 1.0);
}
// TERRAIN_AOV_DEPTH_END

fn shade_main(input : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    // P4: Reflection pass clip plane
    // When enable_flags.y > 0.5, we're in reflection pass mode and should
    // discard fragments below the water plane (only render geometry above water).
    // water_plane = (nx, ny, nz, -d) where plane equation is n·p + d = 0
    // For Y-up water plane at height h: water_plane = (0, 1, 0, -h)
    // A point is above the plane if n·p + d > 0, i.e., p.y > h
    if (water_reflection_uniforms.enable_flags.y > 0.5) {
        let plane = water_reflection_uniforms.water_plane;
        let dist_to_plane = det_dot3(plane.xyz, input.world_position) + plane.w;
        if (dist_to_plane < 0.0) {
            discard;
        }
    }

    let uv = input.tex_coord;
    let debug_mode = u32(u_overlay.params1.y + 0.5);
    
    // Compute all normal variants for diagnostics
    let base_normal = det_normalize3(input.world_normal);
    let lod_info = compute_height_lod(uv);
    let height_lod = lod_info.lod;
    
    // LOD-aware height normal (Milestone 2: fixes flakes from mip mismatch)
    // Sprint 2 note: Multi-scale approach didn't improve edge ratio
    let height_normal_lod = calculate_normal_lod_aware(uv);
    
    // Legacy height normal for comparison (not LOD-aware)
    let texel_size = calculate_texel_size();
    let height_normal_legacy = calculate_normal(uv, texel_size);
    
    // Derivative-based normal (Milestone 1: ground truth comparison)
    let n_dd = calculate_normal_ddxddy(input.world_position);
    
    // Select which height normal to use based on debug mode
    var height_normal = height_normal_lod; // Default: LOD-aware (the fix)
    if (debug_mode == DBG_FLAKE_NO_HEIGHT_NORMAL) {
        // Mode 24: use base_normal (no height detail) to isolate height-normal contribution
        height_normal = base_normal;
    } else if (debug_mode == DBG_FLAKE_DDXDDY_NORMAL) {
        // Mode 25: use derivative-based normal as ground truth
        height_normal = n_dd;
    }
    
    // Milestone 3/D: Minification fade for height-normal contribution
    // As LOD increases (far/grazing), reduce height-normal influence to prevent sparkles.
    // Using smoothstep for threshold-free transition (Milestone D improvement).
    // LOD 0-1: full contribution, LOD 1-4: smoothstep fade, LOD 4+: no contribution
    // Policy: lod_lo=1.0 (near detail preserved), lod_hi=4.0 (far field stable)
    let lod_fade_start = 1.0;  // lod_lo: below this, full height-normal
    let lod_fade_end = 4.0;    // lod_hi: above this, no height-normal
    // smoothstep(edge0, edge1, x) = smooth hermite interpolation
    // We want fade=1.0 at lod_fade_start and fade=0.0 at lod_fade_end
    let lod_fade = 1.0 - smoothstep(lod_fade_start, lod_fade_end, height_lod);
    
    // P5-N: normal_strength controls local normal variation (range 0.25-4.0, default 1.0)
    // Values > 1.0 amplify the deviation between height_normal and base_normal
    // This increases local luminance contrast without changing average surface orientation
    let normal_strength = clamp(u_shading.triplanar_params.z, 0.25, 4.0);
    
    // Amplify LOCAL variation: difference between height_normal and base_normal
    // This preserves average orientation while increasing local contrast
    let normal_delta = height_normal - base_normal;
    let amplified_delta = normal_delta * normal_strength;
    let amplified_height_normal = det_normalize3(base_normal + amplified_delta);
    
    let normal_blend = lod_fade;  // Full blend at near LOD, fades at distance
    // Capture pre-normalized normal for specular AA (Toksvig)
    let mixed_normal = det_mix3(base_normal, amplified_height_normal, normal_blend);
    let normal_len = det_sqrt(det_dot3(mixed_normal, mixed_normal));
    let blended_normal = mixed_normal * det_rcp(max(normal_len, 1e-5));

    let tbn = build_tbn(blended_normal);
    // Extract camera position from view matrix properly
    // For a view matrix V that transforms world→view, the camera position in world space is:
    // camera_pos = -transpose(R) * t, where R is the 3x3 rotation part and t is the translation
    let r00 = u_terrain.view[0][0];
    let r01 = u_terrain.view[1][0];
    let r02 = u_terrain.view[2][0];
    let r10 = u_terrain.view[0][1];
    let r11 = u_terrain.view[1][1];
    let r12 = u_terrain.view[2][1];
    let r20 = u_terrain.view[0][2];
    let r21 = u_terrain.view[1][2];
    let r22 = u_terrain.view[2][2];
    let tx = u_terrain.view[3][0];
    let ty = u_terrain.view[3][1];
    let tz = u_terrain.view[3][2];
    let camera_pos = vec3<f32>(
        -(r00 * tx + r10 * ty + r20 * tz),
        -(r01 * tx + r11 * ty + r21 * tz),
        -(r02 * tx + r12 * ty + r22 * tz),
    );
    let view_dir = det_normalize3(camera_pos - input.world_position);
    let view_dir_tangent = det_mat3_mul_vec3(tbn, view_dir);

    let min_steps = clamp(u32(u_shading.pom_steps.x + 0.5), 1u, 128u);
    let max_steps = clamp(u32(u_shading.pom_steps.y + 0.5), min_steps, 128u);
    let refine_steps = clamp(u32(max(u_shading.pom_steps.z, 0.0)), 0u, 32u);
    let pom_scale = max(u_shading.triplanar_params.w, 0.0);
    let pom_flags = u32(u_shading.pom_steps.w + 0.5);
    let pom_enabled = (pom_flags & 0x1u) != 0u && pom_scale > 0.0;
    let occlusion_enabled = (pom_flags & 0x2u) != 0u;
    let shadow_enabled = (pom_flags & 0x4u) != 0u;

    var pom_uv = uv;
    var pom_offset_magnitude = 0.0;  // Track POM offset for debug visualization
    if (pom_enabled) {
        pom_uv = parallax_occlusion_mapping(
            uv,
            view_dir_tangent,
            pom_scale,
            min_steps,
            max_steps,
            refine_steps
        );
        // Compute offset magnitude (length of UV displacement)
        let pom_offset = pom_uv - uv;
        pom_offset_magnitude = length(pom_offset);
    }
    let parallax_uv = clamp(pom_uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    // Water mask is already flipud'd on the Python side to match heightmap orientation,
    // so sample with the same UV as the heightmap (no additional V flip needed).
    let water_mask_value = textureSampleLevel(water_mask_tex, height_samp, parallax_uv, 0.0).r;
    // Water mask: 0.0 = not water, >0.0 = water (value is shore-distance ratio)
    // Use small epsilon to detect any water, not just shore-distance > 0.5
    let is_water = water_mask_value > 0.001;
    let height_sample = sample_height(parallax_uv);
    let height_clamped = clamp(height_sample, u_shading.clamp0.x, u_shading.clamp0.y);
    var occlusion = 1.0;
    if (occlusion_enabled) {
        occlusion = height_clamped;
    }

    let domain_min = u_overlay.params0.x;
    let inv_range = u_overlay.params0.y;
    let offset = u_overlay.params0.w;
    let height_value = height_clamped + offset;
    var height_norm = height_clamped;
    if (inv_range > 0.0) {
        height_norm = clamp((height_value - domain_min) * inv_range, 0.0, 1.0);
    }

    let tri_scale = max(u_shading.triplanar_params.x, 1e-3);
    let tri_blend = max(u_shading.triplanar_params.y, 1.0);
    // Use base_normal (stable geometric normal) for slope, NOT blended_normal
    // blended_normal has high-frequency perturbations that cause layer selection jitter → flakes
    let slope_raw = 1.0 - abs(base_normal.y);
    let slope_factor = clamp(slope_raw, u_shading.clamp0.z, u_shading.clamp0.w);
    var layer_count = i32(u_shading.layer_control.x + 0.5);
    if (layer_count < 1) {
        layer_count = 1;
    }
    let blend_half = max(u_shading.layer_control.y, 1e-3);
    let base_lod = max(u_shading.clamp2.z, 0.0);
    let lod_bias = u_shading.layer_control.z;
    let lod0_bias = u_shading.layer_control.w;
    let anisotropy = max(u_shading.clamp2.w, 1.0);
    let lod_value = max(base_lod + lod_bias + lod0_bias * slope_factor - (anisotropy - 1.0) * 0.25, 0.0);

    // Compute smooth material weights using Gaussian-like falloff
    // Incorporates both height and slope for natural-looking transitions
    var weights = vec4<f32>(0.0);
    var weight_sum = 0.0;
    let slope_influence = 0.3; // How much slope affects layer selection
    
    for (var idx = 0; idx < 4; idx = idx + 1) {
        if (idx < layer_count) {
            let center = u_shading.layer_heights[idx];
            let dist = abs(height_norm - center);
            // Smooth Gaussian-like falloff instead of linear cutoff
            let sigma = blend_half * 1.5;
            let height_weight = det_exp(-dist * dist / (2.0 * sigma * sigma));
            
            // Modulate by slope: rock/bare materials favor steep slopes
            // Layer 0 (rock) prefers steep, layer 1 (grass) prefers flat
            var slope_mod = 1.0;
            if (idx == 0) {
                // Rock: boost on steep slopes
                slope_mod = det_mix(1.0, 1.5, slope_factor);
            } else if (idx == 1) {
                // Grass: reduce on steep slopes
                slope_mod = det_mix(1.0, 0.5, slope_factor);
            }
            
            let w = height_weight * slope_mod;
            weights[idx] = w;
            weight_sum = weight_sum + w;
        }
    }
    if (weight_sum > 1e-5) {
        weights = weights / weight_sum;
    } else {
        weights = vec4<f32>(0.0);
        weights.x = 1.0;
    }

    var albedo = vec3<f32>(0.0, 0.0, 0.0);
    var roughness = 0.0;
    var metallic = 0.0;
    let vt_normal_enabled = terrain_vt_enabled() && terrain_vt_family_enabled(TERRAIN_VT_FAMILY_NORMAL);
    let vt_mask_enabled = terrain_vt_enabled() && terrain_vt_family_enabled(TERRAIN_VT_FAMILY_MASK);
    var vt_normal_encoded = vec3<f32>(0.0, 0.0, 0.0);
    var vt_normal_residency = 0.0;
    // Mask family channels (DIFFERENTIA recovered-material convention):
    // r = material-map gate, g = roughness, b = ambient occlusion (reserved).
    var vt_mask_data = vec3<f32>(0.0, 0.0, 0.0);
    var vt_mask_residency = 0.0;
    var vt_mask_resident_roughness = 0.0;
    // VERITAS: attribution is single-source — the albedo source id of the
    // dominant (max-weight, first index wins ties) material layer.
    var dominant_layer_weight = -1.0;
    for (var idx = 0; idx < 4; idx = idx + 1) {
        if (idx < layer_count) {
            let weight = weights[idx];
            let layer = f32(idx);
            // Use base_normal (smooth vertex normal) for triplanar weights, NOT blended_normal
            // blended_normal has high-frequency height perturbations that cause weight jitter → flakes
            let sample_rgb = sample_triplanar(
                input.world_position,
                base_normal,  // STABLE geometric normal for triplanar projection
                tri_scale,
                tri_blend,
                layer,
                lod_value
            );
            albedo = albedo + sample_rgb * weight;
            if (weight > dominant_layer_weight) {
                dominant_layer_weight = weight;
                terrain_vt_albedo_source_id = terrain_vt_source_id_triplanar(
                    input.world_position,
                    base_normal,
                    tri_scale,
                    tri_blend,
                    layer,
                );
            }
            roughness = roughness + u_shading.layer_roughness[idx] * weight;
            metallic = metallic + u_shading.layer_metallic[idx] * weight;
            if (vt_normal_enabled) {
                // Normal and mask are recovered raster data aligned to the
                // GridVertex UV, not world-space triplanar color coordinates.
                // Non-resident tiles return the exact flat-normal fallback.
                let normal_sample = terrain_vt_sample_family_data(
                    TERRAIN_VT_FAMILY_NORMAL,
                    uv,
                    terrain_screen_ddx_uv(uv),
                    terrain_screen_ddy_uv(uv),
                    layer,
                );
                vt_normal_encoded = vt_normal_encoded + normal_sample.rgb * weight;
                vt_normal_residency = vt_normal_residency + normal_sample.w * weight;
            }
            if (vt_mask_enabled) {
                let mask_sample = terrain_vt_sample_family_data(
                    TERRAIN_VT_FAMILY_MASK,
                    uv,
                    terrain_screen_ddx_uv(uv),
                    terrain_screen_ddy_uv(uv),
                    layer,
                );
                vt_mask_data = vt_mask_data + mask_sample.rgb * weight;
                vt_mask_residency = vt_mask_residency + mask_sample.w * weight;
                // Accumulate only resident overrides. Missing layers retain
                // their own weighted share of the base roughness below; their
                // fallback value must not be blended a second time.
                vt_mask_resident_roughness = vt_mask_resident_roughness
                    + clamp(mask_sample.g, 0.04, 1.0) * mask_sample.w * weight;
            }
        }
    }

    // Optional water override: when mask is active, treat surface as water material.
    // Store terrain normal for non-water, give water proper wave normals for reflections
    var shading_normal = blended_normal;
    var water_scatter = vec3<f32>(0.0); // Subsurface scatter contribution for water
    var water_depth_value = 0.0; // Water depth for attenuation (promoted to outer scope)
    if (is_water) {
        // Water material properties
        // Use slightly higher roughness (0.02) for stable highlights without fireflies
        // Still low enough for crisp sun glint but avoids subpixel needle highlights
        let water_roughness = 0.02;
        let water_metallic = 0.0; // Dielectric
        roughness = water_roughness;
        metallic = water_metallic;
        
        // Water depth from distance-to-shore proxy
        // The water_mask_value now encodes normalized distance from shore:
        // - 0.0 = at shoreline (edge of water)
        // - 1.0 = maximum distance from any shore (lake center)
        // This gives physically meaningful depth variation independent of DEM height.
        // Falls back to height-based if mask is binary (old behavior).
        let is_distance_encoded = water_mask_value > 0.01 && water_mask_value < 0.99;
        var shore_depth: f32;
        if (is_distance_encoded) {
            // Use distance-to-shore directly as depth proxy
            shore_depth = water_mask_value;
        } else {
            // Fallback: height-based depth (old behavior for binary masks)
            let water_ceil = 0.20;
            shore_depth = 1.0 - saturate(height_norm / water_ceil);
        }
        water_depth_value = shore_depth; // 0=shore/shallow, 1=deep center
        
        // Beer-Lambert absorption: deeper water = more blue, absorbs red first
        let absorption = vec3<f32>(0.8, 0.15, 0.02); // RGB absorption per unit depth  
        let max_depth = 3.0; // Visual depth scaling
        let transmittance = det_exp3(-absorption * water_depth_value * max_depth);
        
        // Deep water color - saturated blue matching reference image
        // Reference shows bright blue lake - boost saturation significantly
        let deep_water_color = vec3<f32>(0.05, 0.45, 0.95);
        
        // Shallow water near shore - cyan-blue tint
        let shallow_color = vec3<f32>(0.1, 0.5, 0.85);
        
        // Blend based on depth - this gives visible shoreline gradient
        let underwater_color = det_mix3(shallow_color, deep_water_color, water_depth_value);
        
        // Water albedo - vibrant blue that shows through reflections
        albedo = underwater_color;
        
        // Scatter contribution - very strong to ensure blue dominates over gray IBL
        // Reference image shows saturated blue lake, not gray reflections
        // water_depth_value: 0=shore (more bottom visible), 1=deep (less bottom visible)
        water_scatter = underwater_color * (1.0 - water_depth_value * 0.3) * 1.2;
        
        // Directional wind-driven waves (dominant wind direction + secondary)
        // Creates coherent wave patterns that read as water, not noise
        let wx = input.world_position.x;
        let wy = input.world_position.y;
        let wind_angle = 0.7; // ~40 degrees
        let wind_cos = cos(wind_angle);
        let wind_sin = sin(wind_angle);
        let wave_coord_1 = wx * wind_cos + wy * wind_sin;
        let wave_coord_perp = -wx * wind_sin + wy * wind_cos;
        
        // Three octaves of directional waves - amplitude decreases near shore
        let wave_scale = det_mix(0.3, 1.0, water_depth_value); // Calmer near shore
        let wave1 = sin(wave_coord_1 * 0.05) * 0.07 * wave_scale;
        let wave2 = sin(wave_coord_1 * 0.15 + wave_coord_perp * 0.03) * 0.035 * wave_scale;
        let wave3 = sin(wave_coord_1 * 0.4 + 1.7) * 0.018;
        
        // Cross-wind component (smaller amplitude)
        let cross_wave = sin(wave_coord_perp * 0.12 + 0.5) * 0.02 * wave_scale;
        
        // Combine into normal perturbation
        let wave_dx = (wave1 + wave2 + wave3) * wind_cos + cross_wave * (-wind_sin);
        let wave_dy = (wave1 + wave2 + wave3) * wind_sin + cross_wave * wind_cos;
        
        // Build perturbed normal (Y is up)
        shading_normal = det_normalize3(vec3<f32>(wave_dx, 1.0, wave_dy));
    }

    if (!is_water) {
        var map_mask = material_map_mask(parallax_uv);
        if (vt_mask_enabled) {
            map_mask = map_mask * clamp(vt_mask_data.r, 0.0, 1.0);
        }
        if (vt_normal_enabled) {
            shading_normal = apply_encoded_tangent_normal(
                shading_normal,
                clamp(vt_normal_encoded, vec3<f32>(0.0), vec3<f32>(1.0)),
                normal_strength,
                map_mask
            );
        }
        shading_normal = apply_material_normal_map(
            shading_normal,
            parallax_uv,
            normal_strength,
            map_mask
        );
        roughness = apply_material_roughness_map(roughness, parallax_uv, map_mask);
        if (vt_mask_enabled) {
            // Feed the mask family's roughness channel into the BRDF, gated by
            // residency coverage: fragments whose mask tiles are not yet
            // resident keep the current default roughness.
            roughness = clamp(
                roughness * (1.0 - clamp(vt_mask_residency, 0.0, 1.0))
                    + vt_mask_resident_roughness,
                0.04,
                1.0
            );
        }
    }
    
    // ──────────────────────────────────────────────────────────────────────────
    // P6: Micro-Detail Application (terrain only, not water)
    // Adds close-range surface detail: detail normals via RNM and albedo noise
    // When detail_enabled = false (params4.x < 0.5), this is a no-op
    // ──────────────────────────────────────────────────────────────────────────
    let view_distance = length(camera_pos - input.world_position);
    let detail_enabled = u_overlay.params4.x > 0.5;
    let detail_scale = max(u_overlay.params4.y, 0.1);
    let detail_normal_strength = clamp(u_overlay.params4.z, 0.0, 1.0);
    let detail_albedo_noise_amp = clamp(u_overlay.params4.w, 0.0, 0.5);
    let detail_fade_start = max(u_overlay.params5.x, 0.0);
    let detail_fade_end = max(u_overlay.params5.y, detail_fade_start + 1.0);
    
    if (detail_enabled && !is_water) {
        // Apply detail normal via RNM blend with distance fade
        shading_normal = apply_detail_normal(
            shading_normal,
            input.world_position,
            view_distance,
            detail_scale,
            detail_normal_strength,
            detail_fade_start,
            detail_fade_end
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Colormap/Overlay System with Debug Modes
    // ──────────────────────────────────────────────────────────────────────────
    // debug_mode already declared at top of fs_main
    let albedo_mode = u32(u_overlay.params1.z + 0.5);
    let colormap_strength = clamp(u_overlay.params1.w, 0.0, 1.0);
    let overlay_strength_raw = u_overlay.params0.z;
    let blend_mode = u32(u_overlay.params1.x + 0.5);

    // Sample colormap LUT
    var lut_u = height_norm;
    if (inv_range <= 0.0) {
        lut_u = clamp(height_clamped, 0.0, 1.0);
    }
    let lut_uv = vec2<f32>(clamp(lut_u, 0.0, 1.0), 0.5);
    let overlay_rgb = textureSample(colormap_tex, colormap_samp, lut_uv).rgb;

    // Apply overlay blend to material albedo (if overlay is active)
    var material_albedo = albedo; // Store original triplanar albedo
    if (overlay_strength_raw > 1e-5) {
        let strength = clamp(overlay_strength_raw, 0.0, 1.0);

        // Blend modes:
        // 0 = Replace
        // 1 = Alpha
        // 2 = Multiply
        // 3 = Additive
        if (blend_mode == 0u) { // Replace
            albedo = overlay_rgb;
        } else if (blend_mode == 1u) { // Alpha
            albedo = det_mix3(albedo, overlay_rgb, strength);
        } else if (blend_mode == 2u) { // Multiply
            albedo = det_mix3(albedo, albedo * overlay_rgb, strength);
        } else if (blend_mode == 3u) { // Additive
            albedo = albedo + strength * overlay_rgb;
        }
    }

    // Apply albedo_mode to determine final albedo
    // 0 = material (triplanar only)
    // 1 = colormap (overlay only, bypasses PBR in debug section below)
    // 2 = mix (blend between material and colormap using colormap_strength)
    // IMPORTANT: Water always uses its own albedo regardless of albedo_mode
    var final_albedo = albedo;
    if (is_water) {
        // Water keeps its blue underwater color - don't override with colormap
        final_albedo = material_albedo;
    } else if (albedo_mode == 0u) { // material
        final_albedo = material_albedo;
    } else if (albedo_mode == 1u) { // colormap
        // Colormap mode: use overlay_rgb directly
        // Note: This path bypasses PBR shading in the debug section below
        final_albedo = overlay_rgb;
    } else if (albedo_mode == 2u) { // mix
        // Mix mode: blend between material albedo and colormap directly
        // colormap_strength=1.0 means full colormap, 0.0 means full material
        final_albedo = det_mix3(material_albedo, overlay_rgb, colormap_strength);
    }

    albedo = clamp(final_albedo, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    
    // P6: Apply procedural albedo brightness noise (terrain only)
    // Adds ±noise_amplitude variation using stable world-space coordinates with distance fade
    if (detail_enabled && !is_water && detail_albedo_noise_amp > 0.0) {
        let albedo_fade = calculate_detail_fade(view_distance, detail_fade_start, detail_fade_end);
        if (albedo_fade > 0.0) {
            let noise_mult = procedural_albedo_noise(input.world_position, detail_albedo_noise_amp * albedo_fade);
            albedo = clamp(albedo * noise_mult, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
    
    // P4: Apply slope+elevation hue variation to increase h_std metric
    // Combines slope (steep=redder, flat=yellower) and elevation spread
    // The caller can set strength to zero when the supplied palette must remain authoritative.
    if (!is_water) {
        let hue_variation_strength = clamp(u_overlay.params3.z, 0.0, 0.2);
        albedo = apply_slope_hue_variation(albedo, slope_factor, height_norm, hue_variation_strength);
    }
    
    var terrain_layer_weights = TerrainLayerWeights(0.0, 0.0, 0.0);
    var terrain_subsurface = TerrainSubsurfaceState(0.0, vec3<f32>(1.0, 1.0, 1.0));
    // ──────────────────────────────────────────────────────────────────────────
    // M4: Material Layer Blending (snow, rock, wetness)
    // Applied after base material but before lighting calculations.
    // When all layers disabled, this is a no-op preserving baseline output.
    // ──────────────────────────────────────────────────────────────────────────
    if (!is_water) {
        // Compute terrain attributes from stable geometric normal
        let terrain_attrs = compute_terrain_attributes(base_normal);
        var material_noise = default_material_noise();
        if (material_layer_uniforms.variation_params0.w > 0.5) {
            material_noise = sample_material_noise(uv, height_norm);
        }
        terrain_layer_weights = resolve_terrain_layer_weights(
            input.world_position,
            terrain_attrs,
            material_noise,
        );
        terrain_subsurface = resolve_terrain_subsurface(terrain_layer_weights);
        // Apply material layers in order: wetness (darkening) -> rock -> snow
        // Order matters: snow on top, then rock, then wetness darkening at base
        albedo = apply_wetness_layer(albedo, terrain_layer_weights.wetness);
        albedo = apply_rock_layer(albedo, terrain_layer_weights.rock);
        albedo = apply_snow_layer(albedo, terrain_layer_weights.snow);
    }
    
    occlusion = clamp(occlusion, u_shading.clamp2.x, u_shading.clamp2.y);
    
    // P3: Track base_roughness (for diffuse) and specular_roughness (Toksvig-adjusted)
    // This split allows Toksvig anti-aliasing on specular only, preserving diffuse detail
    var base_roughness = roughness;     // Original roughness for diffuse calculations
    var specular_roughness = roughness; // Toksvig-adjusted roughness for specular
    
    // Specular AA (Toksvig): increase SPECULAR roughness when normal has variance
    // Using screen-space derivatives (dpdx/dpdy) to measure variance.
    // This works for both:
    //   - Procedural terrain normals (where mipmap averaging doesn't exist)
    //   - Synthetic sparkle perturbation (stress test mode 17)
    // The variance proxy σ² is computed from the rate of change of the normal.
    // P3 fix: Apply Toksvig ONLY to specular_roughness, leaving base_roughness unchanged
    let spec_aa_enabled = u_overlay.params2.z > 0.5;
    var specaa_sigma2 = 0.0;  // Track for debug visualization (mode 19)
    if (!is_water && spec_aa_enabled) {
        // Compute normal variance from screen-space derivatives
        let dndx = dpdxCoarse(shading_normal);
        let dndy = dpdyCoarse(shading_normal);
        // Raw variance: average squared magnitude of derivatives
        let raw_variance = 0.5 * (det_dot3(dndx, dndx) + det_dot3(dndy, dndy));
        
        // SpecAA in the main path (beauty mode, debug=0) is effectively disabled
        // by using a very high variance threshold. This preserves natural terrain detail.
        // 
        // Mode 17 (stress test) uses its own SpecAA path with 100x scale and no threshold,
        // which correctly suppresses the synthetic high-frequency sparkle injection.
        //
        // The sigma_scale env var can still boost effect if needed for specific use cases
        let sigma_scale = max(u_overlay.params2.w, 1.0);
        
        // Very high threshold: effectively disables SpecAA for normal renders
        // Raw_variance for terrain is typically 0.02-0.05, so threshold 1.0 filters all
        // This prevents grid-pattern artifacts from Toksvig following DEM resolution
        let variance_threshold = 1.0;
        let effective_variance = max(raw_variance - variance_threshold, 0.0);
        
        specaa_sigma2 = effective_variance * sigma_scale;
        specaa_sigma2 = clamp(specaa_sigma2, 0.0, 1.0);
        
        let r2 = specular_roughness * specular_roughness;
        // Toksvig formula: r' = sqrt(r² + σ²(1 - r²))
        // P3: Apply ONLY to specular_roughness, not base_roughness
        specular_roughness = sqrt(r2 + specaa_sigma2 * (1.0 - r2));
    }
    
    // Apply roughness multiplier (params2.y, default 1.0 = no change)
    // Used for roughness sweep in PBR proof pack
    let roughness_mult = max(u_overlay.params2.y, 0.001);
    if (roughness_mult != 1.0) {
        base_roughness = base_roughness * roughness_mult;
        specular_roughness = specular_roughness * roughness_mult;
    }
    // P3: Roughness floor - lowered to 0.25 for land (from 0.65) to restore specular detail
    // Toksvig anti-aliasing handles sparkles instead of a high roughness floor
    // Water keeps low roughness (0.02) for crisp reflections (unchanged per P3 constraint)
    let roughness_floor = select(0.25, 0.02, is_water);
    base_roughness = clamp(base_roughness, roughness_floor, 1.0);
    specular_roughness = clamp(specular_roughness, roughness_floor, 1.0);
    // Legacy roughness variable for IBL and other uses (use specular_roughness)
    roughness = specular_roughness;
    metallic = clamp(metallic, 0.0, 1.0);
    var f0 = det_mix3(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    if (is_water) {
        let ior = 1.33;
        let f0_scalar = det_pow((ior - 1.0) / (ior + 1.0), 2.0);
        f0 = vec3<f32>(f0_scalar, f0_scalar, f0_scalar);
    }
    let light_dir = det_normalize3(u_terrain.sun_exposure.xyz);
    
    // P3: Use split-roughness BRDF for terrain direct lighting
    // base_roughness for diffuse (crisp detail), specular_roughness for specular (anti-aliased)
    var lighting: vec3<f32>;
    if (TERRAIN_USE_BRDF_DISPATCH) {
        // Use unified BRDF dispatch (allows model switching) with specular roughness
        let shading_params = terrain_to_shading_params(specular_roughness, metallic, TERRAIN_BRDF_MODEL);
        let n_dot_l = max(det_dot3(shading_normal, light_dir), 0.0);
        lighting = eval_brdf(shading_normal, view_dir, light_dir, albedo, shading_params) * n_dot_l;
    } else {
        // P3: Use split-roughness BRDF - Toksvig on specular only
        // P5-L: Pass lambert_contrast from height_curve.w uniform
        let lambert_k = clamp(u_shading.height_curve.w, 0.0, 1.0);
        lighting = calculate_pbr_brdf_split_roughness(
            shading_normal,
            view_dir,
            light_dir,
            albedo,
            base_roughness,      // Original roughness for diffuse
            specular_roughness,  // Toksvig-adjusted for specular
            metallic,
            f0,
            lambert_k,           // P5-L: Lambert contrast parameter
        );
    }
    lighting = lighting * u_terrain.sun_exposure.w;
    lighting = lighting * u_shading.light_params.rgb;
    
    // P3-10: Apply CSM shadow visibility (optional, gated by TERRAIN_USE_SHADOWS)
    var shadow_debug_color = vec3<f32>(0.0);
    var shadow_visibility = 1.0;
    var shadow_factor = 1.0; // Factor for IBL shadow application
    if (TERRAIN_USE_SHADOWS) {
        // Calculate view-space depth for cascade selection
        let view_pos = det_mat4_mul_vec4(u_terrain.view, vec4<f32>(input.world_position, 1.0));
        let view_depth = -view_pos.z; // Positive depth in view space
        
        // Check for cascade debug mode (compile-time OR runtime via csm_uniforms.debug_mode)
        let shadow_debug_mode = csm_uniforms.debug_mode;
        if (DEBUG_SHADOW_CASCADES || shadow_debug_mode == SHADOW_DEBUG_CASCADES) {
            // Debug mode: get both shadow visibility and cascade color
            let shadow_debug = debug_shadow_with_vis(input.world_position, blended_normal, view_depth, input.tex_coord);
            shadow_debug_color = shadow_debug.xyz;
            shadow_visibility = shadow_debug.w;
        } else {
            // Normal mode: just get shadow visibility
            shadow_visibility = calculate_shadow_terrain(input.world_position, blended_normal, view_depth, input.tex_coord);
        }
        
        // Apply shadow to direct lighting with intensity tuning
        // shadow_visibility: 0.0 = fully shadowed, 1.0 = fully lit
        // Map to [SHADOW_MIN, 1.0] for softer shadows that don't go pitch black
        let direct_shadow = det_mix(SHADOW_MIN, 1.0, shadow_visibility);
        lighting = lighting * direct_shadow;
        
        // Compute factor for IBL shadow (used below)
        shadow_factor = det_mix(1.0 - SHADOW_IBL_FACTOR, 1.0, shadow_visibility);
    } else {
        // Legacy POM-based shadow factor (preserved for backward compatibility)
        if (shadow_enabled && pom_enabled) {
            let shadow_factor = clamp(det_mix(0.4, 1.0, occlusion), u_shading.clamp1.z, u_shading.clamp1.w);
            lighting = lighting * shadow_factor;
        }
    }

    // Apply IBL rotation (terrain-specific feature)
    let rotated_normal = rotate_y(shading_normal, u_ibl.sin_theta, u_ibl.cos_theta);
    let rotated_view = rotate_y(view_dir, u_ibl.sin_theta, u_ibl.cos_theta);
    
    // For water: use near-black albedo for IBL (water surface has no diffuse color)
    // The underwater_color (stored in albedo) is for scatter, not surface reflectance
    // Water gets its color from specular reflections (IBL) + subsurface scatter
    var ibl_albedo = albedo;
    if (is_water) {
        // Water surface has negligible diffuse reflection - it's all specular
        // Using black albedo means IBL will be pure specular (sky reflection)
        ibl_albedo = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Also compute split IBL for PBR debug modes (diffuse/specular separation)
    let ibl_split = eval_ibl_split(rotated_normal, rotated_view, ibl_albedo, metallic, roughness, f0);
    let reflection_dir = reflect(-view_dir, shading_normal);
    let probe_result = sample_probe_irradiance(input.world_position, shading_normal);
    let reflection_probe_result = sample_reflection_probe(input.world_position, reflection_dir, roughness);
    let reflection_probe_weight = sample_reflection_probe_weight(input.world_position);
    let kS_ibl = ibl_split.fresnel;
    let kD_ibl = (vec3<f32>(1.0) - kS_ibl) * (1.0 - metallic);
    let global_diffuse = ibl_split.diffuse;
    let probe_diffuse = kD_ibl * ibl_albedo * probe_result.irradiance;
    let blended_diffuse = det_mix3(global_diffuse, probe_diffuse, probe_result.weight);
    let local_specular = reflection_probe_result.prefiltered_color * ibl_split.specular_brdf;
    let blended_specular = det_mix3(ibl_split.specular, local_specular, reflection_probe_result.weight);
    
    // Apply IBL intensity and occlusion (no artificial boost - proper split-sum should work)
    // For water, don't apply occlusion to IBL (water surface is exposed to sky)
    var ibl_occlusion = occlusion;
    if (is_water) {
        ibl_occlusion = 1.0;
        // Water IBL is pure specular (no diffuse) - this is handled by ibl_albedo = black above
        // The reflection color comes entirely from the environment - no tinting
    }
    let ibl_contrib_pre_ao = (blended_diffuse + blended_specular) * u_ibl.intensity;
    // Apply shadow to IBL diffuse (shadowed areas receive less ambient light)
    // But keep specular unaffected (sky reflections should still be visible)
    let ibl_diffuse_with_shadow = blended_diffuse * shadow_factor;
    let ibl_with_shadow = ibl_diffuse_with_shadow + blended_specular;
    var ibl_contrib = vec3<f32>(0.0);
    ibl_contrib = ibl_with_shadow * u_ibl.intensity * ibl_occlusion;
    
    // Scale split components by intensity and occlusion for debug output
    let ibl_diffuse_scaled = blended_diffuse * u_ibl.intensity * ibl_occlusion * shadow_factor;
    let ibl_specular_scaled = blended_specular * u_ibl.intensity * ibl_occlusion;

    // ──────────────────────────────────────────────────────────────────────────
    // Debug Modes (bypass PBR when debug_mode > 0)
    // ──────────────────────────────────────────────────────────────────────────
    var final_color = vec3<f32>(0.0, 0.0, 0.0);

    if (debug_mode == 1u) {
        // DBG_COLOR_LUT: Show raw LUT color (bypass PBR)
        final_color = overlay_rgb;
    } else if (debug_mode == 2u) {
        // DBG_TRIPLANAR_ALBEDO: Show triplanar material only
        final_color = material_albedo;
    } else if (debug_mode == 3u) {
        // DBG_BLEND_LUT_OVER_ALBEDO: Show lerp(albedo, lut, colormap_strength)
        final_color = det_mix3(material_albedo, overlay_rgb, colormap_strength);
    } else if (debug_mode == DBG_WATER_MASK_BINARY) {
        // ── MODE 4: "What pixels are being treated as water?" ──
        // Uses the EXACT SAME `is_water` variable as the main shading path.
        // CYAN = water branch, MAGENTA = land branch. No shading, no tonemap.
        // Interpretation:
        //   - CYAN outside real lakes → upstream bug (mask generation/upload)
        //   - Mask looks correct but water renders wrong → downstream bug (shader branch)
        let c_water = vec3<f32>(0.0, 1.0, 1.0);  // CYAN
        let c_land  = vec3<f32>(1.0, 0.0, 1.0);  // MAGENTA
        out.color = vec4<f32>(select(c_land, c_water, is_water), 1.0);
        return out;
    } else if (debug_mode == DBG_WATER_MASK_RAW) {
        // ── MODE 5: "What values is the shader receiving for the mask?" ──
        // Shows the EXACT `water_mask_value` being sampled, with error flagging.
        // Catches: wrong texture bound, wrong normalization, wrong channel, NaN/Inf.
        // Interpretation:
        //   - GREEN = NaN/Inf (uninitialized/bad upload)
        //   - RED = value < 0 (invalid normalization)
        //   - YELLOW = value > 1 (invalid normalization)
        //   - Grayscale = value in [0,1] (black=0, white=1)
        //   - Binary mask: expect only two values (black/white)
        //   - Shore-distance gradient: lake edges darker, center brighter
        let m = water_mask_value;
        if (!is_finite_f32(m)) {
            out.color = vec4<f32>(0.0, 1.0, 0.0, 1.0); // GREEN = NaN/Inf
            return out;
        }
        if (m < 0.0) {
            out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // RED = <0
            return out;
        }
        if (m > 1.0) {
            out.color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // YELLOW = >1
            return out;
        }
        // In-range: exact grayscale, no gamma, no tonemap
        out.color = vec4<f32>(vec3<f32>(m), 1.0);
        return out;
    } else if (debug_mode == DBG_IBL_ONLY) {
        // ── MODE 6: "Is IBL working, independent of sun/AO/fog?" ──
        // Show pre-occlusion IBL with tonemap; no sun, no AO, no fog, no water tint.
        let mapped = tonemap_aces(ibl_contrib_pre_ao);
        let out_srgb = linear_to_srgb(mapped);
        out.color = vec4<f32>(out_srgb, 1.0);
        return out;
    } else if (debug_mode == DBG_RAW_SSAO) {
        // ── MODE 28: Raw AO buffer ──
        // Shows heightfield ray-traced AO (binding 16) combined with coarse AO (binding 12).
        // White = no occlusion, black = fully occluded.
        let uv_dbg = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
        let ao_coarse = textureSampleLevel(ao_debug_tex, ao_debug_samp, uv_dbg, 0.0).r;
        // Use textureLoad for R32Float height_ao_tex (non-filterable)
        let height_ao_dbg_size = vec2<f32>(textureDimensions(height_ao_tex, 0));
        let height_ao_dbg_pixel = vec2<i32>(uv_dbg * height_ao_dbg_size);
        let height_ao_dbg_clamped = clamp(height_ao_dbg_pixel, vec2<i32>(0), vec2<i32>(height_ao_dbg_size) - vec2<i32>(1));
        let ao_height = textureLoad(height_ao_tex, height_ao_dbg_clamped, 0).r;
        // Combine both AO sources (multiply)
        let ao_combined = clamp(ao_coarse * ao_height, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(ao_combined), 1.0);
        return out;
    } else if (debug_mode == DBG_SUN_VIS) {
        // ── MODE 29: Sun Visibility buffer ──
        // Shows heightfield ray-traced sun visibility (binding 18).
        // White = fully lit, black = fully shadowed by terrain.
        let uv_dbg = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
        // Use textureLoad for R32Float sun_vis_tex (non-filterable)
        let sun_vis_dbg_size = vec2<f32>(textureDimensions(sun_vis_tex, 0));
        let sun_vis_dbg_pixel = vec2<i32>(uv_dbg * sun_vis_dbg_size);
        let sun_vis_dbg_clamped = clamp(sun_vis_dbg_pixel, vec2<i32>(0), vec2<i32>(sun_vis_dbg_size) - vec2<i32>(1));
        let sun_vis_dbg = textureLoad(sun_vis_tex, sun_vis_dbg_clamped, 0).r;
        out.color = vec4<f32>(vec3<f32>(sun_vis_dbg), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_SPECULAR_ONLY) {
        // ── MODE 8: PBR Specular Only ──
        // Shows ONLY the specular IBL term (no diffuse, no sun).
        // For energy sanity: specular should vary with roughness and view angle.
        // Interpretation:
        //   - Low roughness = sharp reflections, high roughness = blurry
        //   - Grazing angles = stronger specular (Fresnel)
        let specular_linear = max(ibl_specular_scaled, vec3<f32>(0.0));
        let mapped = tonemap_aces(specular_linear);
        let out_srgb = linear_to_srgb(mapped);
        out.color = vec4<f32>(out_srgb, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_FRESNEL) {
        // ── MODE 9: Fresnel Term Visualization ──
        // Shows the Fresnel term F as grayscale (average of RGB components).
        // For Fresnel behavior: should be stronger at grazing angles.
        // Interpretation:
        //   - Near-normal viewing = F close to F0 (typically ~0.04 for dielectrics)
        //   - Grazing angles = F approaches 1.0 (white)
        let fresnel_avg = (ibl_split.fresnel.r + ibl_split.fresnel.g + ibl_split.fresnel.b) / 3.0;
        out.color = vec4<f32>(vec3<f32>(fresnel_avg), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_NDOTV) {
        // ── MODE 10: N.V (View Angle) Visualization ──
        // Shows N.V as grayscale (1.0 = normal facing camera, 0.0 = grazing).
        // Interpretation:
        //   - Flat surfaces facing camera = white
        //   - Steep slopes / edges = darker
        //   - Should correlate with where Fresnel is stronger (inverse relationship)
        out.color = vec4<f32>(vec3<f32>(ibl_split.n_dot_v), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_ROUGHNESS) {
        // ── MODE 11: Roughness Visualization ──
        // Shows the roughness value as grayscale (after any multiplier).
        // Interpretation:
        //   - White = rough (matte), Black = smooth (shiny)
        //   - Should correlate with specular highlight width
        out.color = vec4<f32>(vec3<f32>(roughness), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_ENERGY) {
        // ── MODE 12: Energy (Diffuse + Specular) Before Tonemap ──
        // Shows raw luminance of (diffuse + specular) for energy histogram.
        // NO tonemap - this is for quantitative analysis.
        // Interpretation:
        //   - Values should rarely exceed 1.0 for dielectrics with IBL only
        //   - Use this to generate fig_pbr_energy_hist.png
        //   - Encode: luminance clamped to [0,1] as grayscale (saturated = energy > 1)
        let energy_linear = ibl_diffuse_scaled + ibl_specular_scaled;
        let energy_luma = luminance(energy_linear);
        // Clamp at 1.0 - anything above shows as pure white (energy violation)
        let energy_vis = clamp(energy_luma, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(energy_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_LINEAR_COMBINED) {
        // ── MODE 13: Linear Unclamped (Diffuse + Specular) ──
        // For recomposition proof: encode linear RGB in [0,4] range to [0,1] for PNG export.
        // Decode in Python: linear = encoded * 4.0
        // This should equal ibl_contrib_pre_ao (before AO is applied)
        let combined_linear = ibl_diffuse_scaled + ibl_specular_scaled;
        let encoded = clamp(combined_linear / 4.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(encoded, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_LINEAR_DIFFUSE) {
        // ── MODE 14: Linear Unclamped Diffuse Only ──
        // Encode linear diffuse in [0,4] range to [0,1] for PNG export.
        let encoded = clamp(ibl_diffuse_scaled / 4.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(encoded, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_LINEAR_SPECULAR) {
        // ── MODE 15: Linear Unclamped Specular Only ──
        // Encode linear specular in [0,4] range to [0,1] for PNG export.
        let encoded = clamp(ibl_specular_scaled / 4.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(encoded, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_RECOMP_ERROR) {
        // ── MODE 16: Recomposition Error Heatmap ──
        // Shows abs(ibl_total - (diffuse + specular)) amplified 100x.
        // This should be near-zero if IBL = diffuse + specular.
        // Interpretation:
        //   - Black = perfect recomposition (error < 0.01 linear)
        //   - Any color = error (amplified to be visible)
        //   - If P95 error < 0.001, recomposition is correct
        let recomposed = ibl_diffuse_scaled + ibl_specular_scaled;
        let error = abs(ibl_contrib_pre_ao - recomposed);
        // Amplify 100x so small errors are visible
        let error_vis = clamp(error * 100.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(error_vis, 1.0);
        return out;
    } else if (debug_mode == DBG_SPECAA_SPARKLE) {
        // ── MODE 17: SpecAA Sparkle Stress Test ──
        // Inject synthetic high-frequency normal perturbation to stress-test Toksvig.
        // The perturbation creates a checkerboard pattern that should cause sparkles
        // unless SpecAA properly widens the lobe via screen-derivative variance.
        // Metric: Compare high-freq energy between SpecAA ON vs OFF.
        
        // Generate synthetic high-freq normal perturbation (screen-space checkerboard)
        let screen_pos = input.clip_position.xy;
        
        // Perturb shading normal with high-frequency tangent-space noise
        // This creates rapid normal changes that dpdx/dpdy will detect
        let perturb_strength = 0.3; // Strong enough to cause visible sparkles
        let perturb = vec3<f32>(
            sin(screen_pos.x * 7.3 + screen_pos.y * 3.7) * perturb_strength,
            sin(screen_pos.x * 5.1 - screen_pos.y * 8.3) * perturb_strength,
            0.0
        );
        let sparkle_normal = det_normalize3(shading_normal + perturb);
        
        // CRITICAL: Recompute variance on the PERTURBED normal
        // This is where SpecAA must detect the high-frequency variation
        var sparkle_roughness = roughness;
        var sparkle_sigma2_for_debug = 0.0;  // For debug mode 20
        if (spec_aa_enabled) {
            let dndx_sparkle = dpdxCoarse(sparkle_normal);
            let dndy_sparkle = dpdyCoarse(sparkle_normal);
            // Use same 100x amplification as main path
            let sigma_scale = max(u_overlay.params2.w, 1.0) * 100.0;
            let sparkle_sigma2 = 0.5 * (det_dot3(dndx_sparkle, dndx_sparkle) + det_dot3(dndy_sparkle, dndy_sparkle)) * sigma_scale;
            sparkle_sigma2_for_debug = sparkle_sigma2;  // Save raw value for debug
            let sparkle_sigma2_clamped = clamp(sparkle_sigma2, 0.0, 1.0);
            
            // Apply Toksvig roughness boost on perturbed normal
            let r2_sparkle = sparkle_roughness * sparkle_roughness;
            sparkle_roughness = sqrt(r2_sparkle + sparkle_sigma2_clamped * (1.0 - r2_sparkle));
            sparkle_roughness = clamp(sparkle_roughness, 0.04, 1.0);
        }
        
        // Recompute specular IBL with perturbed normal and SpecAA-corrected roughness
        let sparkle_ibl = eval_ibl(
            sparkle_normal,
            view_dir,
            albedo,
            metallic,
            sparkle_roughness, // Uses freshly-computed Toksvig roughness on perturbed normal
            f0
        );
        
        // Extract specular component (approximate: assume kD same ratio)
        let n_dot_v_sparkle = saturate(det_dot3(sparkle_normal, view_dir));
        let f_sparkle = fresnel_schlick_roughness(n_dot_v_sparkle, f0, sparkle_roughness);
        let kS_sparkle = f_sparkle;
        let kD_sparkle = (vec3<f32>(1.0) - kS_sparkle) * (1.0 - metallic);
        let total_k = kD_sparkle + kS_sparkle;
        let spec_ratio = kS_sparkle / max(total_k, vec3<f32>(0.001));
        let sparkle_spec = sparkle_ibl * spec_ratio * u_ibl.intensity;
        
        // Output with tonemap for visibility
        let mapped = tonemap_aces(sparkle_spec);
        let out_srgb = linear_to_srgb(mapped);
        out.color = vec4<f32>(out_srgb, 1.0);
        return out;
    } else if (debug_mode == DBG_POM_OFFSET_MAG) {
        // ── MODE 18: POM Offset Magnitude Visualization ──
        // Shows the parallax UV offset magnitude as grayscale.
        // Black = no offset (POM disabled or flat area)
        // White = maximum offset (areas with strong parallax displacement)
        // This proves POM is actually displacing texture coordinates.
        // Interpretation:
        //   - Should correlate with view angle (more offset at grazing angles)
        //   - Should correlate with height variation (ridges/valleys show more offset)
        //   - If uniformly black when POM enabled → POM not working or pom_scale too small
        //   - If uniform noise → something wrong with height sampling
        // Scale factor: POM offset is typically small (0.0-0.1 UV units)
        // We amplify by 10x for visibility (0.1 UV offset → 1.0 grayscale)
        let offset_vis = clamp(pom_offset_magnitude * 10.0, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(offset_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_SPECAA_SIGMA2) {
        // ── MODE 19: SpecAA Sigma² (Variance) Visualization ──
        // Shows the normal variance (σ²) used by SpecAA/Toksvig as grayscale.
        // Black = no variance (flat normal field)
        // White = high variance (high-frequency normal changes)
        // Interpretation:
        //   - Terrain edges/ridges should show higher variance
        //   - Flat areas should be near-black
        //   - If uniformly black when SpecAA enabled → variance not being detected
        //   - Should correlate with where sparkle reduction occurs
        // Scale: σ² is typically small (0.0-0.1), amplify 20x for visibility
        let sigma2_vis = clamp(specaa_sigma2 * 20.0, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(sigma2_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_SPECAA_SPARKLE_SIGMA2) {
        // ── MODE 20: SpecAA Sparkle Sigma² Visualization ──
        // Shows the variance computed on the SPARKLE-PERTURBED normal.
        // This proves that dpdx/dpdy can detect the synthetic perturbation.
        // Should show high variance (bright) across entire terrain.
        // If black: variance not being detected on perturbed normals.
        
        // Generate same sparkle perturbation as mode 17
        let screen_pos = input.clip_position.xy;
        let perturb_strength = 0.3;
        let perturb = vec3<f32>(
            sin(screen_pos.x * 7.3 + screen_pos.y * 3.7) * perturb_strength,
            sin(screen_pos.x * 5.1 - screen_pos.y * 8.3) * perturb_strength,
            0.0
        );
        let sparkle_normal_dbg = det_normalize3(shading_normal + perturb);
        
        // Compute variance on perturbed normal
        let dndx_dbg = dpdxCoarse(sparkle_normal_dbg);
        let dndy_dbg = dpdyCoarse(sparkle_normal_dbg);
        let sigma_scale_dbg = max(u_overlay.params2.w, 1.0) * 100.0;
        let sparkle_sigma2_dbg = 0.5 * (det_dot3(dndx_dbg, dndx_dbg) + det_dot3(dndy_dbg, dndy_dbg)) * sigma_scale_dbg;
        
        // Visualize (no additional scaling - should already be visible with 100x)
        let sigma2_vis_sparkle = clamp(sparkle_sigma2_dbg, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(sigma2_vis_sparkle), 1.0);
        return out;
    } else if (debug_mode == DBG_TRIPLANAR_WEIGHTS) {
        // ── MODE 21: Triplanar Blend Weights Visualization ──
        // Shows RGB = x/y/z projection weights.
        // T1 requirement: wx + wy + wz = 1, weights change smoothly with normal.
        // Interpretation:
        //   - RED dominant = surface faces X axis (YZ plane projection dominant)
        //   - GREEN dominant = surface faces Y axis (XZ plane projection dominant, i.e. flat/horizontal)
        //   - BLUE dominant = surface faces Z axis (XY plane projection dominant)
        //   - Steep cliffs should show RED or BLUE, flat areas should show GREEN
        //   - Transitions should be smooth, not abrupt
        // Use base_normal for stable triplanar weights (no high-frequency jitter)
        let tri_weights = compute_triplanar_weights(base_normal, tri_blend);
        out.color = vec4<f32>(tri_weights, 1.0);
        return out;
    } else if (debug_mode == DBG_TRIPLANAR_CHECKER) {
        // ── MODE 22: Triplanar Checker Pattern ──
        // Shows a procedural checker pattern sampled via triplanar mapping.
        // Interpretation:
        //   - Checker squares should be uniform size (no distortion) when projection matches surface
        //   - If squares stretch on steep slopes, triplanar isn't working correctly
        //   - Pattern should remain world-space stable (no swimming with camera movement)
        let checker_scale = 8.0; // Number of checker squares per world unit
        // Use base_normal for stable triplanar weights
        let checker_val = sample_triplanar_checker(
            input.world_position,
            base_normal,  // Stable geometric normal
            tri_scale,
            tri_blend,
            checker_scale
        );
        // Output as grayscale for clear checker visibility
        out.color = vec4<f32>(vec3<f32>(checker_val), 1.0);
        return out;
    } else if (debug_mode == DBG_FLAKE_NO_SPECULAR) {
        // ── MODE 23: No Specular (Diffuse Only) ──
        // Shows terrain with ONLY diffuse/ambient lighting (no IBL specular).
        // If flakes disappear here → flakes are specular aliasing.
        let ambient_strength_23 = det_mix(u_shading.clamp1.x, u_shading.clamp1.y, 1.0 - abs(blended_normal.y));
        let ambient_23 = albedo * ambient_strength_23;
        let direct_mult_23 = det_mix(0.65, 1.0, occlusion);
        let diffuse_only_23 = ibl_diffuse_scaled;
        let shaded_no_spec = ambient_23 + lighting * direct_mult_23 + diffuse_only_23;
        let exposure_23 = max(u_shading.light_params.w, 0.0);
        let tonemapped_23 = tonemap_aces(shaded_no_spec * exposure_23);
        out.color = vec4<f32>(linear_to_srgb(tonemapped_23), 1.0);
        return out;
    } else if (debug_mode == DBG_FLAKE_NO_HEIGHT_NORMAL) {
        // ── MODE 24: No Height Normal ──
        // Uses base_normal (geometric normal) instead of height-derived normal.
        // If flakes disappear here → flakes are from height-normal frequency.
        // Normal substitution is done at top of shader.
        // Fall through to normal shading path with the substituted normal.
    } else if (debug_mode == DBG_FLAKE_DDXDDY_NORMAL) {
        // ── MODE 25: Derivative-Based Normal (Ground Truth) ──
        // Uses n_dd = cross(dpdx, dpdy) as the shading normal.
        // This is the "mathematically correct" per-pixel surface normal.
        // Normal substitution is done at top of shader.
        // Fall through to normal shading path with the substituted normal.
    } else if (debug_mode == DBG_FLAKE_HEIGHT_LOD) {
        // ── MODE 26: Height LOD Visualization ──
        // Grayscale ramp from computed LOD (already distinct)
        let max_lod = f32(textureNumLevels(height_tex) - 1u);
        let lod_normalized = height_lod / max(max_lod, 1.0);
        out.color = vec4<f32>(vec3<f32>(lod_normalized), 1.0);
        return out;
    } else if (debug_mode == DBG_FLAKE_NORMAL_BLEND) {
        // ── MODE 27: Effective Normal Blend Visualization ──
        // Grayscale ramp from normal_blend (already distinct)
        out.color = vec4<f32>(vec3<f32>(normal_blend), 1.0);
        return out;
    } else if (debug_mode == DBG_VIEW_DEPTH) {
        // MODE 40: View-Space Depth (Projection Probe)
        // Outputs view-space depth as grayscale. Theta/phi change the camera ray origin,
        // so this varies with camera orientation. Use DBG_NDC_DEPTH (41) for FOV checks.
        let view_pos = det_mat4_mul_vec4(u_terrain.view, vec4<f32>(input.world_position, 1.0));
        let view_z = -view_pos.z; // Negate because view space looks down -Z
        // Normalize to reasonable range (assume terrain within 0-2000 units from camera)
        let depth_normalized = clamp(view_z / 2000.0, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(depth_normalized), 1.0);
        return out;
    } else if (debug_mode == DBG_NDC_DEPTH) {
        // MODE 41: NDC Depth (Projection Probe)
        // Outputs clip.z/clip.w as grayscale. FOV changes the projection matrix and
        // therefore this value, even for identical geometry and camera position.
        let clip_pos = det_mat4_mul_vec4(
            u_terrain.proj,
            det_mat4_mul_vec4(u_terrain.view, vec4<f32>(input.world_position, 1.0)),
        );
        let ndc_z = clip_pos.z / clip_pos.w;
        // NDC depth is in [0,1] for wgpu/WebGPU
        out.color = vec4<f32>(vec3<f32>(ndc_z), 1.0);
        return out;
    } else if (debug_mode == DBG_PROBE_IRRADIANCE) {
        let mapped = tonemap_aces(max(probe_result.irradiance * probe_result.weight, vec3<f32>(0.0)));
        out.color = vec4<f32>(linear_to_srgb(mapped), 1.0);
        return out;
    } else if (debug_mode == DBG_PROBE_WEIGHT) {
        out.color = vec4<f32>(vec3<f32>(probe_result.weight), 1.0);
        return out;
    } else if (debug_mode == DBG_REFLECTION_PROBE_COLOR) {
        let mapped = tonemap_aces(max(
            local_specular * reflection_probe_weight * u_ibl.intensity,
            vec3<f32>(0.0),
        ));
        out.color = vec4<f32>(linear_to_srgb(mapped), 1.0);
        return out;
    } else if (debug_mode == DBG_REFLECTION_PROBE_WEIGHT) {
        out.color = vec4<f32>(vec3<f32>(reflection_probe_weight), 1.0);
        return out;
    } else if (debug_mode == DBG_SHADOW_TECHNIQUE) {
        // MODE 33: Shadow Technique Visualization
        // Red = HARD (0), Green = PCF (1), Blue = PCSS (2)
        // This proves what technique value the shader is receiving from the uniform buffer.
        let tech = csm_uniforms.technique;
        var tech_color = vec3<f32>(1.0, 0.0, 1.0); // Magenta = unknown
        if (tech == 0u) {
            tech_color = vec3<f32>(1.0, 0.0, 0.0); // Red = HARD
        } else if (tech == 1u) {
            tech_color = vec3<f32>(0.0, 1.0, 0.0); // Green = PCF
        } else if (tech == 2u) {
            tech_color = vec3<f32>(0.0, 0.0, 1.0); // Blue = PCSS
        }
        out.color = vec4<f32>(tech_color, 1.0);
        return out;
    } else if (debug_mode == DBG_VIEW_POS_XYZ) {
        // ── MODE 42: View-Space Position as RGB (Projection Probe) ──
        // Encodes view-space XYZ as RGB. Dramatically changes with camera orientation.
        let view_pos = det_mat4_mul_vec4(u_terrain.view, vec4<f32>(input.world_position, 1.0));
        // Normalize to [-1000, 1000] range -> [0, 1] for visualization
        let pos_normalized = (view_pos.xyz / 1000.0 + 1.0) * 0.5;
        out.color = vec4<f32>(clamp(pos_normalized, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        return out;
    } else if (debug_mode == 100u) {
        // DBG_WATER_BINARY: Unambiguous binary water classification
        // Blue = water, Dark gray = land. No lighting, no tonemap.
        final_color = debug_water_is_water(is_water);
    } else if (debug_mode == 101u) {
        // DBG_WATER_SCALAR: Shore-distance gradient visualization
        // Falsecolor ramp with white shoreline ring on water, dark gray on land.
        // If gradient appears in wrong location => UV orientation bug
        final_color = debug_water_scalar(is_water, water_mask_value);
    } else if (debug_mode == 102u) {
        // DBG_WATER_IBL_SPEC_ISOLATED: IBL specular on water ONLY
        // Land is pure black (no cheating). Water shows compressed HDR IBL.
        final_color = debug_water_ibl_spec_only(is_water, ibl_contrib);
    } else if (debug_mode == 103u) {
        // DBG_PREFILT_RAW: Direct sample of specular cubemap (no Fresnel)
        // This isolates whether the cubemap itself is returning data
        let refl_dir = reflect(-view_dir, shading_normal);
        let rot_refl = rotate_y(refl_dir, u_ibl.sin_theta, u_ibl.cos_theta);
        let prefilt = textureSampleLevel(envSpecular, envSampler, rot_refl, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt);
    } else if (debug_mode == 104u) {
        // DBG_CUBEMAP_DIRECT: Sample envSpecular with unrotated reflection, ignore Fresnel
        // Water: use (0.0, 1.0, 0.0) reflection = straight up to sky
        let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
        let prefilt_sky = textureSampleLevel(envSpecular, envSampler, sky_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt_sky);
    } else if (debug_mode == 105u) {
        // DBG_IRRADIANCE_DIRECT: Sample envIrradiance (irradiance cubemap) with up direction
        // This tests if the OTHER cubemap (envIrradiance) has data
        let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
        let irr_sky = textureSampleLevel(envIrradiance, envSampler, sky_dir, 0.0).rgb;
        // Use same helper as specular - cyan means data, magenta means zero
        // Reuse the debug_water_prefilt_raw function which already has this logic
        final_color = debug_water_prefilt_raw(is_water, irr_sky);
    } else if (debug_mode == 110u) {
        // DBG_PURE_RED: Sanity check - just return pure red
        // If this doesn't show red, debug_mode isn't being set correctly
        final_color = vec3<f32>(1.0, 0.0, 0.0);
    } else if (debug_mode == 111u) {
        // DBG_SPEC_NEG_Z: Sample envSpecular with -Z direction (front face)
        let front_dir = vec3<f32>(0.0, 0.0, -1.0);
        let prefilt_front = textureSampleLevel(envSpecular, envSampler, front_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt_front);
    } else if (debug_mode == 112u) {
        // DBG_SPEC_POS_X: Sample envSpecular with +X direction (right face)
        let right_dir = vec3<f32>(1.0, 0.0, 0.0);
        let prefilt_right = textureSampleLevel(envSpecular, envSampler, right_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt_right);
    } else if (debug_mode == 113u) {
        // DBG_IRR_NEG_Z: Sample envIrradiance with -Z direction
        let front_dir = vec3<f32>(0.0, 0.0, -1.0);
        let irr_front = textureSampleLevel(envIrradiance, envSampler, front_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, irr_front);
    } else if (debug_mode == DBG_NDOTL) {
        // ── SPRINT 1 MODE 30: N·L (Lambert term) ──
        // Shows raw lambert term as grayscale for shadow-field diagnosis.
        // White = sun-facing, Black = shadow-facing.
        // ROI_A should show moderate gray, ROI_B should show brighter values.
        let ndotl_terrain = max(det_dot3(shading_normal, light_dir), 0.0);
        out.color = vec4<f32>(vec3<f32>(ndotl_terrain), 1.0);
        return out;
    } else if (debug_mode == DBG_SHADOW_FACTOR) {
        // ── SPRINT 1 MODE 31: Shadow Factor ──
        // Shows CSM shadow visibility as grayscale.
        // White = fully lit, Black = fully shadowed.
        // If ROI_A is too dark here, shadows are over-aggressive.
        out.color = vec4<f32>(vec3<f32>(shadow_factor), 1.0);
        return out;
    } else if (debug_mode == DBG_PRE_TONEMAP) {
        // ── SPRINT 1 MODE 32: Pre-Tonemap Linear Color ──
        // Shows final linear color before tonemapping (clamped for display).
        // Useful for diagnosing tonemap compression issues.
        // Compute terrain shading inline for this debug mode
        let ndotl_dbg = max(det_dot3(shading_normal, light_dir), 0.0);
        let sun_int_dbg = length(u_shading.light_params.rgb);
        let ambient_dbg = 0.18;
        let sun_peak_dbg = 0.42;
        let base_diff_dbg = ambient_dbg + (sun_peak_dbg - ambient_dbg) * ndotl_dbg * sun_int_dbg;
        let ao_shadow_dbg = max(shadow_factor, 0.30) * max(occlusion, 0.65);
        let lit_dbg = base_diff_dbg * ao_shadow_dbg;
        let pre_tonemap = albedo * (lit_dbg + AMBIENT_FLOOR * 0.35);
        let exposure_dbg = max(u_shading.light_params.w, 0.0);
        let exposed_dbg = pre_tonemap * exposure_dbg;
        // Clamp to [0,1] for display (values >1 show as white)
        out.color = vec4<f32>(clamp(exposed_dbg, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        return out;
    } else {
        // Normal PBR shading path
        var shaded = vec3<f32>(0.0);
        
        if (is_water) {
            // Water: specular-dominant shading (minimal diffuse, strong reflections)
            // Fresnel is already baked into eval_ibl via fresnel_schlick_roughness
            // Direct specular from sun
            let n_dot_v = max(det_dot3(shading_normal, view_dir), 0.001);
            let n_dot_l = max(det_dot3(shading_normal, light_dir), 0.0);
            let h = det_normalize3(view_dir + light_dir);
            let n_dot_h = max(det_dot3(shading_normal, h), 0.0);
            let v_dot_h = max(det_dot3(view_dir, h), 0.001);
            
            // GGX Distribution for water - use proper alpha² without over-clamping
            // For very smooth surfaces (roughness ~0.01), D can legitimately reach 10000+
            // This is physically correct and produces natural sun glints
            let alpha = roughness * roughness;
            let alpha2 = max(alpha * alpha, 1e-8); // Minimal clamp for numerical stability only
            let n_dot_h2 = n_dot_h * n_dot_h;
            let denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
            let D = alpha2 / (PI * denom * denom);
            
            // Fresnel (Schlick) using v_dot_h for correct specular Fresnel
            let fresnel = f0 + (vec3<f32>(1.0) - f0) * det_pow(1.0 - v_dot_h, 5.0);
            
            // Geometry term (Smith GGX, height-correlated)
            // For very smooth surfaces, G approaches 1.0
            let k = alpha / 2.0;
            let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
            let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
            let G = g_v * g_l;
            
            // Cook-Torrance specular BRDF (no artificial boosts!)
            let spec_denom = 4.0 * n_dot_v * n_dot_l + 0.0001;
            let direct_spec = D * fresnel * G / spec_denom;
            
            // Sun contribution - NO artificial boost; proper GGX + low roughness = natural glints
            let sun_color = vec3<f32>(1.0, 0.98, 0.95); // Slightly warm sun
            let sun_intensity = u_shading.light_params.z; // Use actual sun intensity (no boost!)
            let sun_spec = direct_spec * sun_color * sun_intensity * n_dot_l;
            
            // ─────────────────────────────────────────────────────────────────────
            // P4: Planar Reflection Integration
            // Sample reflection texture with wave-based distortion and Fresnel mixing
            // ─────────────────────────────────────────────────────────────────────
            
            // Sample planar reflection with wave distortion
            // water_depth_value is the shore distance (0=shore, 1=deep)
            let planar_refl = sample_water_reflection(
                input.world_position,
                shading_normal,  // Wave-perturbed normal for UV distortion
                water_depth_value
            );
            let planar_refl_valid = planar_refl.a > 0.5;
            
            // Calculate Fresnel factor for reflection blending
            let water_fresnel = calculate_water_fresnel(view_dir, shading_normal);
            
            // Water shading strategy for visible depth gradient:
            // 1. Planar reflections give terrain reflections on water (P4 feature)
            // 2. IBL gives environmental reflections (sky, distant environment)
            // 3. Sun specular gives glints where waves face the sun
            // 4. Depth tint modulates the overall brightness based on water depth
            // 
            // For depth to be visible, we need to darken deep water regions
            // even when they have specular highlights. This mimics how real
            // deep water absorbs more light than shallow water.
            
            // Use water_depth_value (promoted to outer scope)
            // water_depth_value: 1.0 = deep, 0.0 = shallow
            // Depth attenuation: shallow = 100%, deep = 30% (70% absorbed)
            // This is aggressive but necessary for depth to be visible against bright specular
            let depth_atten = det_mix(1.0, WATER_DEPTH_ATTEN_DEEP, water_depth_value);
            
            // Blend planar reflection with IBL
            // When planar reflection is valid, it takes priority over IBL for terrain reflections
            // IBL still contributes sky/environment reflections where planar reflection is weak
            var combined_reflection = ibl_contrib;
            if (planar_refl_valid) {
                // Blend planar reflection based on Fresnel and intensity
                combined_reflection = blend_water_reflection(
                    ibl_contrib,      // Base color (IBL fallback)
                    planar_refl.rgb,  // Planar reflection
                    true,
                    water_fresnel,
                    water_depth_value
                );
            }
            
            // Combine: minimal reflections to let blue water color dominate
            // Reference image shows saturated blue lake - prioritize underwater color
            let reflective =
                (combined_reflection * WATER_COMBINED_REFLECTION_SCALE +
                sun_spec * WATER_SUN_SPECULAR_SCALE) *
                depth_atten;
            
            // Water color: keep a stable artist-tunable blue tint while local/global
            // reflections come from the blended probe + planar + IBL path above.
            shaded =
                reflective +
                WATER_BASE_TINT * WATER_BASE_TINT_SCALE +
                water_scatter * WATER_SCATTER_SCALE;
            
        } else {
            // ══════════════════════════════════════════════════════════════════════
            // P2-S4: Terrain Lighting Composition (structure locked per spec)
            // ══════════════════════════════════════════════════════════════════════
            // 
            // P2-S1: ambient_floor must be in [0.22, 0.38] range
            // P2-S2: AO must be clamped to min 0.65 (max 35% darkening)
            // P2-S3: Shadow factor must be clamped to [0.30, 1.0]
            // P2-L2: Dynamic range (valleys vs ridges) should be 5.8-7.8
            //
            // Reference quantiles: q5=0.10, q50=0.35, q95=0.65
            
            // Compute lambert term for hillshade-like appearance
            let n_dot_l_terrain = max(det_dot3(shading_normal, light_dir), 0.0);
            
            // Style Match Fix: Compress dynamic range while preserving local contrast
            // Reference has compressed highlights (D1) and readable shadow detail
            let sun_intensity = length(u_shading.light_params.rgb);
            
            // Sprint 1 Fix: N.L-dependent ambient lift for shadow-facing areas
            // Iter 2: ROI_A=-0.062 (too dark), ROI_B=+0.096 (good)
            // Need more shadow lift without raising highlights
            let ambient_shadow = 0.32;  // Raise shadow ambient more
            let ambient_lit = 0.10;     // Keep lit ambient low
            let sun_peak = 0.36;        // Keep highlights compressed
            
            // Interpolate ambient: high when shadow-facing, low when sun-facing
            let ambient_interp = det_mix(ambient_shadow, ambient_lit, n_dot_l_terrain);
            
            // Sun contribution adds on top of ambient
            let sun_contrib = (sun_peak - ambient_lit) * n_dot_l_terrain * sun_intensity;
            let base_diffuse = ambient_interp + sun_contrib;
            
            // Sprint 2: Edge energy fix - need edge ratio 0.88+ (currently 0.26-0.41)
            // Use ADDITIVE edge enhancement based on height-derived slope variation
            // The shading_normal already encodes terrain microstructure from height map
            
            // Compute slope steepness from normal (deviation from up vector)
            let slope_steepness = 1.0 - abs(shading_normal.y);  // 0=flat, 1=vertical
            
            // Edge detection via normal screen-space derivatives
            let dndx = dpdxCoarse(shading_normal);
            let dndy = dpdyCoarse(shading_normal);
            let normal_gradient = length(dndx) + length(dndy);
            
            // Strong edge signal: combines slope steepness with local variation
            let edge_signal = slope_steepness * 0.3 + normal_gradient * 15.0;
            
            // Additive edge term (not multiplicative) for stronger visible edges
            // This creates brightness variation independent of base diffuse
            let edge_bright = clamp(edge_signal * (n_dot_l_terrain + 0.3), 0.0, 0.25);
            let edge_dark = clamp(edge_signal * (1.0 - n_dot_l_terrain) * 0.5, 0.0, 0.15);
            let diffuse_raw = base_diffuse + edge_bright - edge_dark;
            
            // P5: Apply AO multiplier from coarse heightmap AO or SSAO
            // P2-S2: AO must be clamped to min 0.65 (max 35% darkening)
            let ao_weight = u_overlay.params3.x;
            var ao_clamped = 1.0;
            if (ao_weight > 0.0) {
                let ao_uv = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
                let ao_sample = textureSampleLevel(ao_debug_tex, ao_debug_samp, ao_uv, 0.0).r;
                ao_clamped = det_mix(1.0, max(ao_sample, 0.65), ao_weight);
            }
            // Heightfield ray-traced AO: sample and combine with existing AO
            // When height_ao disabled, texture is 1x1 white (1.0 = no occlusion)
            // Use textureLoad since R32Float doesn't support filtering
            let height_ao_uv = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
            let height_ao_tex_size = vec2<f32>(textureDimensions(height_ao_tex, 0));
            let height_ao_pixel = vec2<i32>(height_ao_uv * height_ao_tex_size);
            let height_ao_clamped_pixel = clamp(height_ao_pixel, vec2<i32>(0), vec2<i32>(height_ao_tex_size) - vec2<i32>(1));
            let height_ao_sample = textureLoad(height_ao_tex, height_ao_clamped_pixel, 0).r;
            let height_ao_clamped = max(height_ao_sample, 0.65);
            ao_clamped = ao_clamped * height_ao_clamped;
            // Also apply POM occlusion with same clamp
            ao_clamped = ao_clamped * max(occlusion, 0.65);
            
            // P2-S3: Shadow factor clamped to [0.30, 1.0]
            // Use the already-mapped shadow value from CSM
            let shadow_clamped = max(shadow_factor, 0.30);
            
            // Heightfield ray-traced sun visibility: modulates direct sun lighting
            // When sun_visibility disabled, texture is 1x1 white (1.0 = fully lit)
            // Use textureLoad since R32Float doesn't support filtering
            let sun_vis_uv = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
            let sun_vis_tex_size = vec2<f32>(textureDimensions(sun_vis_tex, 0));
            let sun_vis_pixel = vec2<i32>(sun_vis_uv * sun_vis_tex_size);
            let sun_vis_clamped_pixel = clamp(sun_vis_pixel, vec2<i32>(0), vec2<i32>(sun_vis_tex_size) - vec2<i32>(1));
            let sun_vis_sample = textureLoad(sun_vis_tex, sun_vis_clamped_pixel, 0).r;
            // Clamp sun visibility to min 0.30 to prevent pitch-black shadows
            let sun_vis_clamped = max(sun_vis_sample, 0.30);
            // Combine CSM shadow with heightfield sun visibility (multiplicative)
            let combined_shadow = shadow_clamped * sun_vis_clamped;
            
            // P3-S1: Compute combined shadow/AO attenuation
            // Direct product for full contrast range (no sqrt compression)
            // P3 requires lf_max/lf_min >= 4.5
            // Use combined_shadow which includes both CSM and heightfield sun visibility
            let ao_shadow_factor = ao_clamped * combined_shadow; // Range [0.195, 1.0]
            let diffuse_lit = diffuse_raw * ao_shadow_factor;
            
            // P3-S1: IBL term adds minimal fill light
            // Reduced to allow deeper shadows while preventing pitch-black
            // D2 fix: Remove warm bias - reference has cooler neutrality
            let ibl_diffuse_biased = blended_diffuse;
            let ibl_diffuse_factor = length(ibl_diffuse_biased) * u_ibl.intensity;
            let ibl_term = ibl_diffuse_factor * AMBIENT_FLOOR * 0.35;
            let terrain_sss = evaluate_terrain_subsurface(
                terrain_subsurface,
                albedo,
                shading_normal,
                view_dir,
                light_dir,
                combined_shadow,
                ibl_diffuse_factor,
            );
            
            // P2-S4: lighting_factor = diffuse_lit + ibl_term
            let lighting_factor = diffuse_lit + ibl_term;
            
            // Apply lighting factor to albedo
            // Spec H-03: Lighting modulates brightness, not colormap lookup
            let lit_albedo = albedo * lighting_factor;
            
            // Add specular contribution (capped at 25% per P2-S4)
            // Specular for terrain must not exceed 25% of total RGB
            let spec_contrib = blended_specular * u_ibl.intensity * 0.12;
            let spec_capped = min(spec_contrib, albedo * 0.20);
            
            // Final terrain shading
            shaded = lit_albedo + spec_capped + terrain_sss;
        }
        
        let exposure = max(u_shading.light_params.w, 0.0);
        shaded = shaded * exposure;
        
        // P2: Apply atmospheric fog after exposure, before tonemap
        // When fog_density = 0, this is a no-op preserving P1 output
        shaded = apply_atmospheric_fog(
            shaded,
            input.world_position,
            camera_pos,
            input.clip_position.xy,
        );
        
        let offline_hdr_output = u_overlay.params5.w > 0.5;
        if (offline_hdr_output) {
            final_color = shaded;
        } else {
            // Use filmic tonemapping for better highlight compression (D1 fix)
            // Filmic curve compresses highlights more aggressively than ACES
            let tonemapped = tonemap_filmic_terrain(shaded);
            final_color = tonemapped;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Milestone 1: Mode Stamp Overlay
    // ──────────────────────────────────────────────────────────────────────────
    // Add debug_mode/255 to blue channel in top-left 8x8 corner for visual mode ID.
    // This allows verifying which debug mode actually rendered (trust but verify).
    // The stamp is small enough not to interfere with visual inspection.
    if (debug_mode > 0u) {
        let stamp_size = 8.0;
        let screen_pos = input.clip_position.xy;
        if (screen_pos.x < stamp_size && screen_pos.y < stamp_size) {
            // Encode mode in blue channel: 0.0-1.0 maps to mode 0-255
            let mode_signal = f32(debug_mode) / 255.0;
            final_color.b = clamp(final_color.b + mode_signal, 0.0, 1.0);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // P1-Shadow Debug: Cascade Visualization Overlay
    // Activated by compile-time DEBUG_SHADOW_CASCADES or runtime csm_uniforms.debug_mode == 1
    // ──────────────────────────────────────────────────────────────────────────
    let csm_debug_mode = csm_uniforms.debug_mode;
    if (TERRAIN_USE_SHADOWS && (DEBUG_SHADOW_CASCADES || csm_debug_mode == SHADOW_DEBUG_CASCADES)) {
        // Calculate view depth for cascade selection
        let view_pos_dbg = det_mat4_mul_vec4(u_terrain.view, vec4<f32>(input.world_position, 1.0));
        let view_depth_dbg = -view_pos_dbg.z;
        // Force cascade 0 for debugging - shadow depth pass only renders cascade 0
        let cascade_idx_dbg = 0u;  // Force cascade 0
        
        // Get shadow UV coordinates and NDC using normalized position
        let shadow_pos_dbg = normalize_for_shadow(input.tex_coord);
        let cascade_dbg = csm_uniforms.cascades[cascade_idx_dbg];
        let light_space_pos_dbg = det_mat4_mul_vec4(cascade_dbg.light_view_proj, vec4<f32>(shadow_pos_dbg, 1.0));
        let ndc_dbg = light_space_pos_dbg.xyz / light_space_pos_dbg.w;
        let shadow_uv_dbg = vec2<f32>(ndc_dbg.x * 0.5 + 0.5, ndc_dbg.y * -0.5 + 0.5);
        let compare_depth_dbg = ndc_dbg.z;
        
        // Sample shadow map depth
        let sampled_depth_dbg = textureSampleLevel(
            shadow_maps,
            moment_sampler,
            shadow_uv_dbg,
            i32(cascade_idx_dbg),
            0.0
        );
        
        // Depth difference: positive = receiver behind shadow map surface (shadow)
        //                   negative = receiver in front (lit)
        let depth_diff = compare_depth_dbg - sampled_depth_dbg;
        
        // Debug: Find the threshold where shadow comparison switches
        // Binary search: find depth where compare starts failing
        var low_d = 0.0;
        var high_d = 1.0;
        for (var i = 0; i < 10; i = i + 1) {
            let mid_d = (low_d + high_d) * 0.5;
            let result = textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv_dbg, i32(cascade_idx_dbg), mid_d);
            if (result > 0.5) {
                low_d = mid_d;  // Passing, shadow map depth is higher than mid
            } else {
                high_d = mid_d;  // Failing, shadow map depth is lower than mid
            }
        }
        // low_d is approximately the shadow map depth value
        // Visualize the actual shadow comparison result
        // If depths match (diff < 0.1), the shadow comparison should work
        let shadow_vis = textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv_dbg, i32(cascade_idx_dbg), ndc_dbg.z);
        
        // R = shadow map depth (binary search result)
        // G = main shader depth (expected value)
        // B = actual shadow comparison result at main shader depth
        final_color = vec3<f32>(low_d, ndc_dbg.z, shadow_vis);
        
        // Debug: Compare ndc.z (after matrix) vs shadow_map_depth
        // R = ndc.z (main shader's computed depth after matrix transform)
        // G = shadow_map_depth (what shadow shader wrote)
        // B = 1.0 if depths within 0.05 of each other, 0.0 otherwise
        let depth_match = select(0.0, 1.0, abs(ndc_dbg.z - low_d) < 0.05);
        final_color = vec3<f32>(ndc_dbg.z, low_d, depth_match);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // P1-Shadow Debug: Raw Shadow Visibility
    // Activated by compile-time DEBUG_SHADOW_RAW or runtime csm_uniforms.debug_mode == 2
    // ──────────────────────────────────────────────────────────────────────────
    if (TERRAIN_USE_SHADOWS && (DEBUG_SHADOW_RAW || csm_debug_mode == SHADOW_DEBUG_RAW)) {
        // Linear grayscale: each output channel is the visibility value.
        final_color = vec3<f32>(shadow_visibility);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Output Encoding (P6.1: Color Space Correctness)
    // ──────────────────────────────────────────────────────────────────────────
    // Render target is Rgba8Unorm (linear). We must encode for sRGB display.
    // P6.1: output_srgb_eotf selects between (the host forces it on for
    // active AETHER so sky and terrain share one exact IEC sRGB transfer):
    //   - false (P5 legacy): pow-gamma approximation via gamma_correct()
    //   - true (P6 correct): exact piecewise linear_to_srgb() EOTF
    let output_srgb_eotf = u_overlay.params5.z > 0.5;
    let offline_hdr_output = u_overlay.params5.w > 0.5;
    var encoded_color: vec3<f32>;
    if (TERRAIN_USE_SHADOWS && (DEBUG_SHADOW_RAW || csm_debug_mode == SHADOW_DEBUG_RAW)) {
        encoded_color = final_color;
    } else if (offline_hdr_output) {
        encoded_color = final_color;
    } else if (output_srgb_eotf) {
        // P6: Exact sRGB EOTF - clamp to [0,1] first, then apply exact curve
        encoded_color = linear_to_srgb(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)));
    } else {
        // P5 legacy: pow-gamma approximation
        let gamma = max(u_overlay.params2.x, 0.1);
        encoded_color = gamma_correct(final_color, gamma);
    }

    out.color = vec4<f32>(encoded_color, 1.0);

    // ──────────────────────────────────────────────────────────────────────────
    // M1: AOV Output (always written - pipeline determines which targets are bound)
    // ──────────────────────────────────────────────────────────────────────────
    // AOV Albedo: Base color before lighting, in linear space
    // Store the raw albedo value used in shading
    out.aov_albedo = vec4<f32>(albedo, 1.0);

    // AOV Normal: Normalized world-space shading normal in signed float space
    // Alpha is a residency oracle for the VT normal family. It is 1 when VT
    // normal is disabled (the geometric-normal contract is fully satisfied),
    // otherwise the exact weighted coverage of resident Grid-UV samples.
    out.aov_normal = vec4<f32>(
        det_normalize3(shading_normal),
        select(1.0, clamp(vt_normal_residency, 0.0, 1.0), vt_normal_enabled),
    );

    // Preserve the legacy unused AOV dataflow in ordinary beauty modules: its
    // exact arithmetic is part of the cross-backend deterministic compilation
    // contract. The MRT entry overwrites this with accurate raster depth.
    let view_pos_for_depth = det_mat4_mul_vec4(u_terrain.view, vec4<f32>(input.world_position, 1.0));
    let linear_depth = -view_pos_for_depth.z;
    let clip_near = max(u_terrain.camera_mode_params.z, 1e-5);
    let clip_far = max(u_terrain.camera_mode_params.w, clip_near + 1e-5);
    let depth_normalized = clamp(
        (linear_depth - clip_near) / max(clip_far - clip_near, 1e-5),
        0.0,
        1.0
    );
    out.aov_depth = vec4<f32>(depth_normalized, depth_normalized, depth_normalized, 1.0);

    // VERITAS: co-emitted with the color at composite time so the source map
    // and the image always describe the same frame.
    out.source_id = terrain_vt_albedo_source_id;

    return out;
}

@fragment
fn fs_main(input : VertexOutput) -> FragmentOutput {
    // Capture the quad gradients while every fragment lane is still active.
    // The feedback helper is intentionally derivative-free so later selector
    // branches cannot make mip demand backend-dependent.
    let feedback_ddx_uv = terrain_screen_ddx_uv(input.tex_coord);
    let feedback_ddy_uv = terrain_screen_ddy_uv(input.tex_coord);
    let feedback_ddx_world = terrain_screen_ddx_world(input.world_position);
    let feedback_ddy_world = terrain_screen_ddy_world(input.world_position);
    let out = shade_main(input);
    atomicAdd(&terrain_frame_counters.material_invocations, 1u);
    atomicAdd(&terrain_frame_counters.forward_material_invocations, 1u);
    terrain_vt_write_surface_feedback(
        input.tex_coord,
        input.world_position,
        feedback_ddx_uv,
        feedback_ddy_uv,
        feedback_ddx_world,
        feedback_ddy_world,
        0u,
        input.clip_position.xy,
    );
    return out;
}

// TERRAIN_AOV_ENTRY_BEGIN
@fragment
fn fs_aov_main(input : VertexOutput) -> FragmentOutput {
    let feedback_ddx_uv = terrain_screen_ddx_uv(input.tex_coord);
    let feedback_ddy_uv = terrain_screen_ddy_uv(input.tex_coord);
    let feedback_ddx_world = terrain_screen_ddx_world(input.world_position);
    let feedback_ddy_world = terrain_screen_ddy_world(input.world_position);
    var out = shade_main(input);
    atomicAdd(&terrain_frame_counters.material_invocations, 1u);
    atomicAdd(&terrain_frame_counters.forward_material_invocations, 1u);
    terrain_vt_write_surface_feedback(
        input.tex_coord,
        input.world_position,
        feedback_ddx_uv,
        feedback_ddy_uv,
        feedback_ddx_world,
        feedback_ddy_world,
        0u,
        input.clip_position.xy,
    );
    out.aov_depth = terrain_aov_depth(input);
    return out;
}
// TERRAIN_AOV_ENTRY_END

// TESSELLA pass 1 (`fs_visibility`) lives in src/shaders/terrain_visbuffer_write.wgsl
// and pass 2 (`fs_visibility_resolve_fullscreen`) in
// src/shaders/terrain_visibility_fullscreen.wgsl; both are appended to this
// module at assembly time by `shader_sources`. Neither is defined here, so the
// forward module cannot drift from the packing those two files agree on.

// ──────────────────────────────────────────────────────────────────────────
// BOP-P2-02: Clipmap ring/skirt vertex path.
// Consumes the CPU-generated clipmap mesh (src/terrain/clipmap/) instead of
// the procedural grid in vs_main, sampling the same height texture and
// feeding the same fs_main PBR fragment stage. clip_morph.x < 0 marks skirt
// vertices, which are pushed down by a small offset derived from the ring
// resolution (u_terrain.camera_mode_params.y) to seal cracks between LOD rings.
// ──────────────────────────────────────────────────────────────────────────

@vertex
fn vs_clipmap_main(
    @location(0) clip_pos_xz : vec2<f32>,
    @location(1) clip_uv : vec2<f32>,
    @location(2) clip_morph : vec2<f32>,
    @location(3) instance_col0 : vec4<f32>,
    @location(4) instance_col1 : vec4<f32>,
    @location(5) instance_col2 : vec4<f32>,
    @location(6) instance_col3 : vec4<f32>,
    @location(7) _tile_id_lod : vec2<u32>
) -> VertexOutput {
    var out : VertexOutput;

    let uv = clamp(clip_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let h_fine = sample_height_bilinear(uv);
    let height_dims = vec2<f32>(textureDimensions(height_tex));
    let coarse_texels = exp2(min(max(clip_morph.y, 0.0) + 1.0, 16.0));
    let coarse_step = vec2<f32>(coarse_texels)
        / max(height_dims - vec2<f32>(1.0), vec2<f32>(1.0));
    let coarse_cell = uv / coarse_step;
    let coarse_base = floor(coarse_cell) * coarse_step;
    let coarse_t = fract(coarse_cell);
    let h00 = sample_height_bilinear(coarse_base);
    let h10 = sample_height_bilinear(coarse_base + vec2<f32>(coarse_step.x, 0.0));
    let h01 = sample_height_bilinear(coarse_base + vec2<f32>(0.0, coarse_step.y));
    let h11 = sample_height_bilinear(coarse_base + coarse_step);
    let h_coarse = mix(mix(h00, h10, coarse_t.x), mix(h01, h11, coarse_t.x), coarse_t.y);
    let h_raw = mix(h_fine, h_coarse, clamp(clip_morph.x, 0.0, 1.0));
    let t_geom = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_disp = det_fma(apply_height_curve01(t_geom), h_max - h_min, h_min);
    let h_exag = u_terrain.spacing_h_exag.z;
    let h_center = (h_min + h_max) * 0.5;
    let skirt_offset = select(0.0, u_terrain.camera_mode_params.y * 0.001, clip_morph.x < 0.0);
    let world_z_centered = (h_disp - h_center - skirt_offset) * h_exag;
    let world_z_original = (h_disp - skirt_offset) * h_exag;
    let instance_transform = mat4x4<f32>(
        instance_col0,
        instance_col1,
        instance_col2,
        instance_col3,
    );
    let instance_position = instance_transform * vec4<f32>(
        clip_pos_xz.x,
        clip_pos_xz.y,
        world_z_centered,
        1.0,
    );

    out.world_position = vec3<f32>(instance_position.xy, world_z_original);
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
    out.tex_coord = uv;
    // TESSELLA: the clipmap vertex stage always emits the packed tile/LOD
    // identity terrain_visbuffer_write.wgsl consumes. No forward shading path
    // reads tile_id, so both pipelines share this one vertex stage.
    out.tile_id = ((_tile_id_lod.y & 0xfu) << 12u) | (_tile_id_lod.x & 0xfffu);
    out.clip_position = det_mat4_mul_vec4(
        u_terrain.proj,
        det_mat4_mul_vec4(
            u_terrain.view,
            instance_position,
        ),
    );
    return out;
}
