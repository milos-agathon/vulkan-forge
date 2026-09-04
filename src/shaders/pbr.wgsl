//! PBR (Physically-Based Rendering)
//! Bind Groups and Layouts:
//! - @group(0): Per-draw uniforms
//!   - @binding(0): uniform buffer `Uniforms`
//!   - @binding(1): uniform buffer `PbrLighting`
//!   - @binding(2): uniform buffer `ShadingParamsGPU` (BRDF dispatch, defined in lighting.wgsl)
//! - @group(1): Material textures/samplers (if used)
//!   - @binding(0): uniform buffer `PbrMaterial`
//!   - @binding(1): sampled texture (base color)
//!   - @binding(2): sampled texture (metallic-roughness)
//!   - @binding(3): sampled texture (normal)
//!   - @binding(4): sampled texture (occlusion)
//!   - @binding(5): sampled texture (emissive)
//!   - @binding(6): sampler (filtering)
//! - @group(2): Shadow maps (P3-08)
//!   - @binding(0): uniform buffer `CsmUniforms`
//!   - @binding(1): texture_depth_2d_array (shadow maps)
//!   - @binding(2): sampler_comparison (shadow sampler)
//!   - @binding(3): texture_2d_array<f32> (moment maps)
//!   - @binding(4): sampler (moment sampler)
//! - @group(3): IBL textures (optional)
//!   - @binding(0): irradiance texture
//!   - @binding(1): irradiance sampler
//!   - @binding(2): prefilter texture
//!   - @binding(3): prefilter sampler
//!   - @binding(4): BRDF LUT texture
//!   - @binding(5): BRDF LUT sampler
//! Render Target Formats:
//! - Color: RGBA8UnormSrgb (the target applies final sRGB encoding)
//! Address Space: `uniform`, `fragment`, `vertex`
//!
//! Implements metallic-roughness workflow with BRDF dispatch to lighting.wgsl module.

// Import centralized lighting and BRDF definitions
#include "lighting.wgsl"
// Import unified IBL evaluator (P4 spec: group(2) bindings)
#include "lighting_ibl.wgsl"
// Note: shadows.wgsl is concatenated before this file in pipeline/pbr.rs

struct Uniforms {
    model_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct PbrMaterial {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    emissive: vec3<f32>,
    alpha_cutoff: f32,
    texture_flags: u32,
    // Implicit padding to 64 bytes total (matches CPU struct)
    // See docs/pbr_cpu_gpu_alignment.md for details
}

struct PbrLighting {
    light_direction: vec3<f32>,
    _padding1: f32,                  // Explicit padding for std140 alignment
    light_color: vec3<f32>,
    light_intensity: f32,
    camera_position: vec3<f32>,
    _padding2: f32,                  // Explicit padding for std140 alignment
    ibl_intensity: f32,
    ibl_rotation: f32,
    exposure: f32,
    gamma: f32,
}

// Note: ShadingParamsGPU and BRDF constants are defined in lighting.wgsl

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
}

// Uniforms and textures
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> lighting: PbrLighting;
@group(0) @binding(2) var<uniform> shading: ShadingParamsGPU;

@group(1) @binding(0) var<uniform> material: PbrMaterial;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;
@group(1) @binding(6) var material_sampler: sampler;

// Shadow resources (P3-08) - group(2) matches shadows.wgsl
// Note: Shadow bindings are defined in shadows.wgsl, included above

// IBL textures - group(2) as per P4 spec (unified bindings in lighting_ibl.wgsl)
// Note: Bindings are declared in lighting_ibl.wgsl:
// @group(2) @binding(0) envSpecular : texture_cube<f32>
// @group(2) @binding(1) envIrradiance : texture_cube<f32>
// @group(2) @binding(2) envSampler : sampler
// @group(2) @binding(3) brdfLUT : texture_2d<f32>

// Texture flags
const FLAG_BASE_COLOR: u32 = 1u;
const FLAG_METALLIC_ROUGHNESS: u32 = 2u;
const FLAG_NORMAL: u32 = 4u;
const FLAG_OCCLUSION: u32 = 8u;
const FLAG_EMISSIVE: u32 = 16u;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform position to world space then to clip space
    let world_position = uniforms.model_matrix * vec4<f32>(input.position, 1.0);
    output.world_position = world_position.xyz;
    output.clip_position = uniforms.projection_matrix * uniforms.view_matrix * world_position;
    
    // Pass through UV coordinates
    output.uv = input.uv;
    
    // Transform TBN vectors to world space
    let normal_mat = mat3x3<f32>(
        uniforms.normal_matrix[0].xyz,
        uniforms.normal_matrix[1].xyz,
        uniforms.normal_matrix[2].xyz
    );
    
    output.world_normal = normalize(normal_mat * input.normal);
    output.world_tangent = normalize(normal_mat * input.tangent);
    output.world_bitangent = normalize(normal_mat * input.bitangent);
    
    return output;
}

// Sample normal map and transform to world space
fn sample_normal_map(uv: vec2<f32>, tbn: mat3x3<f32>) -> vec3<f32> {
    if (material.texture_flags & FLAG_NORMAL) != 0u {
        let normal_sample = textureSample(normal_texture, material_sampler, uv);
        
        // Decode normal from texture (range [0,1] to [-1,1])
        var tangent_normal = normal_sample.xyz * 2.0 - 1.0;
        
        // Apply normal scale
        tangent_normal = vec3<f32>(
            tangent_normal.xy * material.normal_scale,
            tangent_normal.z
        );
        
        // Normalize to ensure unit length
        tangent_normal = normalize(tangent_normal);
        
        // Transform to world space
        return normalize(tbn * tangent_normal);
    } else {
        // Use vertex normal
        return normalize(tbn[2]);
    }
}

// Note: Common BRDF math (distribution_ggx, geometry_smith, fresnel_schlick) now imported from brdf/common.wgsl via lighting.wgsl

// Fresnel-Schlick with roughness moved to lighting_ibl.wgsl (used by eval_ibl)

// IBL sampling functions removed - now using eval_ibl from lighting_ibl.wgsl
// Old 2D equirectangular sampling replaced with cubemap + LUT (P4 spec requirement)

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Construct TBN matrix
    let T = normalize(input.world_tangent);
    let B = normalize(input.world_bitangent);
    let N = normalize(input.world_normal);
    let TBN = mat3x3<f32>(T, B, N);
    
    // Sample normal map and get world normal
    let world_normal = sample_normal_map(input.uv, TBN);
    
    // Sample material textures
    var base_color = material.base_color;
    if (material.texture_flags & FLAG_BASE_COLOR) != 0u {
        base_color = base_color * textureSample(base_color_texture, material_sampler, input.uv);
    }
    
    var metallic = material.metallic;
    var roughness = material.roughness;
    if (material.texture_flags & FLAG_METALLIC_ROUGHNESS) != 0u {
        let mr_sample = textureSample(metallic_roughness_texture, material_sampler, input.uv);
        metallic = metallic * mr_sample.b; // Blue channel = metallic
        roughness = roughness * mr_sample.g; // Green channel = roughness
    }
    
    // Clamp roughness to avoid singularities
    roughness = clamp(roughness, 0.04, 1.0);
    
    var occlusion = 1.0;
    if (material.texture_flags & FLAG_OCCLUSION) != 0u {
        occlusion = mix(1.0, textureSample(occlusion_texture, material_sampler, input.uv).r, material.occlusion_strength);
    }
    
    var emissive = material.emissive;
    if (material.texture_flags & FLAG_EMISSIVE) != 0u {
        emissive = emissive * textureSample(emissive_texture, material_sampler, input.uv).rgb;
    }
    
    // Alpha testing
    if base_color.a < material.alpha_cutoff {
        discard;
    }
    
    // Calculate lighting vectors
    let view_dir = normalize(lighting.camera_position - input.world_position);
    let light_dir = normalize(-lighting.light_direction);
    let half_dir = normalize(light_dir + view_dir);
    let reflection_dir = reflect(-view_dir, world_normal);
    
    // Calculate dot products
    let n_dot_v = max(dot(world_normal, view_dir), 0.0);
    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let n_dot_h = max(dot(world_normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    
    // Calculate F0 (surface reflection at zero incidence)
    let dielectric_f0 = vec3<f32>(0.04);
    let f0 = mix(dielectric_f0, base_color.rgb, metallic);
    
    // DIRECT LIGHTING (P2-03: BRDF dispatch via eval_brdf, P3-08: shadow visibility)
    var direct_lighting = vec3<f32>(0.0);
    if n_dot_l > 0.0 {
        // Construct ShadingParamsGPU from material properties and shading uniform
        // Use shading.brdf for model selection, and material metallic/roughness for surface properties
        var shading_params: ShadingParamsGPU;
        shading_params.brdf = shading.brdf;
        shading_params.metallic = metallic;
        shading_params.roughness = roughness;
        shading_params.sheen = shading.sheen;
        shading_params.clearcoat = shading.clearcoat;
        shading_params.subsurface = shading.subsurface;
        shading_params.anisotropy = shading.anisotropy;
        
        // Call unified BRDF dispatch
        let brdf_color = eval_brdf(world_normal, view_dir, light_dir, base_color.rgb, shading_params);
        let radiance = lighting.light_color * lighting.light_intensity;
        
        // P3-08: Apply shadow visibility
        // Calculate view-space depth for cascade selection
        let view_pos = uniforms.view_matrix * vec4<f32>(input.world_position, 1.0);
        let view_depth = -view_pos.z; // Positive depth in view space
        
        // Sample shadows (returns 0.0 = full shadow, 1.0 = no shadow)
        let shadow_visibility =
            calculate_shadow(input.world_position, view_depth, world_normal);
        
        // Apply shadow to direct lighting only (IBL unaffected)
        direct_lighting = brdf_color * radiance * n_dot_l * shadow_visibility;
    }
    
    // INDIRECT LIGHTING (IBL) - using unified eval_ibl from lighting_ibl.wgsl
    // Conceptual split used later by GI composition:
    //   L_diffuse_base = direct diffuse (from eval_brdf) + diffuse IBL
    //   L_spec_base    = direct specular   (from eval_brdf) + specular IBL
    // Here both lobes are already combined into a single RGB `indirect_lighting` term.
    var indirect_lighting = vec3<f32>(0.0);
    
    // Check if we have IBL enabled
    let has_ibl = lighting.ibl_intensity > 0.0;
    
    if has_ibl {
        // eval_ibl returns diffuse + specular IBL in linear HDR units
        indirect_lighting = eval_ibl(world_normal, view_dir, base_color.rgb, metallic, roughness, f0);
        indirect_lighting = indirect_lighting * lighting.ibl_intensity;
    } else {
        // Simple ambient lighting fallback (treated as part of L_diffuse_base)
        let ambient = vec3<f32>(0.03) * base_color.rgb;
        indirect_lighting = ambient;
    }
    
    // Apply ambient occlusion to indirect (diffuse) term. Specular is unaffected here; any
    // future GI composition pass must ensure AO only modulates the diffuse component.
    indirect_lighting = indirect_lighting * occlusion;
    
    // At this stage the fragment output color can be viewed as:
    //   color = L_diffuse_base + L_spec_base + emissive
    // where direct_lighting contains the BRDF-evaluated direct diffuse+spec, and
    // indirect_lighting contains the diffuse+spec IBL contribution.
    var color = direct_lighting + indirect_lighting + emissive;
    
    // Tone mapping. The Rgba8UnormSrgb target applies final sRGB encoding.
    color = color * lighting.exposure;
    color = tonemap_reinhard(color);
    
    return vec4<f32>(color, base_color.a);
}

// Simplified PBR fragment shader without IBL (for fallback)
@fragment
fn fs_pbr_simple(input: VertexOutput) -> @location(0) vec4<f32> {
    // Construct TBN matrix
    let T = normalize(input.world_tangent);
    let B = normalize(input.world_bitangent);
    let N = normalize(input.world_normal);
    let TBN = mat3x3<f32>(T, B, N);
    
    // Sample normal map
    let world_normal = sample_normal_map(input.uv, TBN);
    
    // Sample material properties
    var base_color = material.base_color;
    if (material.texture_flags & FLAG_BASE_COLOR) != 0u {
        base_color = base_color * textureSample(base_color_texture, material_sampler, input.uv);
    }
    
    var metallic = material.metallic;
    var roughness = clamp(material.roughness, 0.04, 1.0);
    
    if (material.texture_flags & FLAG_METALLIC_ROUGHNESS) != 0u {
        let mr_sample = textureSample(metallic_roughness_texture, material_sampler, input.uv);
        metallic = metallic * mr_sample.b;
        roughness = clamp(roughness * mr_sample.g, 0.04, 1.0);
    }
    
    var emissive = material.emissive;
    if (material.texture_flags & FLAG_EMISSIVE) != 0u {
        emissive = emissive * textureSample(emissive_texture, material_sampler, input.uv).rgb;
    }
    
    // Alpha testing
    if base_color.a < material.alpha_cutoff {
        discard;
    }
    
    // Lighting calculation
    let view_dir = normalize(lighting.camera_position - input.world_position);
    let light_dir = normalize(-lighting.light_direction);
    let half_dir = normalize(light_dir + view_dir);
    
    let n_dot_v = max(dot(world_normal, view_dir), 0.0);
    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let n_dot_h = max(dot(world_normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    
    // Calculate F0
    let f0 = mix(vec3<f32>(0.04), base_color.rgb, metallic);
    
    // BRDF calculation (P2-03: using eval_brdf, P3-08: shadow visibility)
    var color = vec3<f32>(0.0);
    
    if n_dot_l > 0.0 {
        // Construct shading params using current shading uniform
        var shading_params: ShadingParamsGPU;
        shading_params.brdf = shading.brdf;
        shading_params.metallic = metallic;
        shading_params.roughness = roughness;
        shading_params.sheen = shading.sheen;
        shading_params.clearcoat = shading.clearcoat;
        shading_params.subsurface = shading.subsurface;
        shading_params.anisotropy = shading.anisotropy;
        
        let brdf_color = eval_brdf(world_normal, view_dir, light_dir, base_color.rgb, shading_params);
        let radiance = lighting.light_color * lighting.light_intensity;
        
        // P3-08: Apply shadow visibility
        let view_pos = uniforms.view_matrix * vec4<f32>(input.world_position, 1.0);
        let view_depth = -view_pos.z;
        let shadow_visibility =
            calculate_shadow(input.world_position, view_depth, world_normal);
        
        color = brdf_color * radiance * n_dot_l * shadow_visibility;
    }
    
    // Simple ambient
    color = color + vec3<f32>(0.03) * base_color.rgb + emissive;
    
    // Tone mapping. The Rgba8UnormSrgb target applies final sRGB encoding.
    color = color * lighting.exposure;
    color = tonemap_reinhard(color);
    
    return vec4<f32>(color, base_color.a);
}

// Debug fragment shaders for development
@fragment
fn fs_debug_normals(input: VertexOutput) -> @location(0) vec4<f32> {
    let T = normalize(input.world_tangent);
    let B = normalize(input.world_bitangent);
    let N = normalize(input.world_normal);
    let TBN = mat3x3<f32>(T, B, N);
    
    let world_normal = sample_normal_map(input.uv, TBN);
    return vec4<f32>(world_normal * 0.5 + 0.5, 1.0);
}

@fragment
fn fs_debug_metallic_roughness(input: VertexOutput) -> @location(0) vec4<f32> {
    var metallic = material.metallic;
    var roughness = material.roughness;
    
    if (material.texture_flags & FLAG_METALLIC_ROUGHNESS) != 0u {
        let mr_sample = textureSample(metallic_roughness_texture, material_sampler, input.uv);
        metallic = metallic * mr_sample.b;
        roughness = roughness * mr_sample.g;
    }
    
    return vec4<f32>(roughness, metallic, 0.0, 1.0);
}

@fragment
fn fs_debug_base_color(input: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = material.base_color;
    if (material.texture_flags & FLAG_BASE_COLOR) != 0u {
        base_color = base_color * textureSample(base_color_texture, material_sampler, input.uv);
    }
    
    return vec4<f32>(base_color.rgb, 1.0);
}
