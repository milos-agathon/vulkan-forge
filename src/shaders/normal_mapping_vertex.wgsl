//! Vertex and fragment shaders for normal mapping pipeline
//! 
//! Implements tangent-space normal mapping with proper TBN transformation
//! and lighting calculations in world space.

struct Uniforms {
    model_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    light_direction: vec4<f32>, // w component for strength
    normal_strength: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

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

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0) 
var normal_texture: texture_2d<f32>;

@group(1) @binding(1)
var normal_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform position to world space
    let world_position = uniforms.model_matrix * vec4<f32>(input.position, 1.0);
    output.world_position = world_position.xyz;
    
    // Transform to clip space
    let view_position = uniforms.view_matrix * world_position;
    output.clip_position = uniforms.projection_matrix * view_position;
    
    // Transform TBN vectors to world space using normal matrix
    let normal_mat = mat3x3<f32>(
        uniforms.normal_matrix[0].xyz,
        uniforms.normal_matrix[1].xyz, 
        uniforms.normal_matrix[2].xyz
    );
    
    output.world_normal = normalize(normal_mat * input.normal);
    output.world_tangent = normalize(normal_mat * input.tangent);
    output.world_bitangent = normalize(normal_mat * input.bitangent);
    
    // Pass through UV coordinates
    output.uv = input.uv;
    
    return output;
}

/// Decode normal from texture sample and convert to tangent space
fn decode_normal_map(normal_sample: vec4<f32>, strength: f32) -> vec3<f32> {
    // Decode from [0,1] texture range to [-1,1] normal range
    var tangent_normal = normal_sample.rgb * 2.0 - 1.0;
    
    // Apply strength/intensity scaling
    tangent_normal = vec3<f32>(tangent_normal.xy * strength, tangent_normal.z);
    
    // Ensure Z component maintains unit length constraint
    let xy_len_sq = dot(tangent_normal.xy, tangent_normal.xy);
    tangent_normal.z = sqrt(max(0.0, 1.0 - xy_len_sq));
    
    return tangent_normal;
}

/// Transform a tangent-space normal to world space using TBN matrix
fn apply_normal_map(
    tangent_normal: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>, 
    normal: vec3<f32>
) -> vec3<f32> {
    // Construct TBN matrix (tangent -> world space transform)
    let tbn = mat3x3<f32>(
        tangent,
        bitangent,
        normal
    );
    
    // Transform tangent-space normal to world space
    let world_normal = tbn * tangent_normal;
    
    // Normalize to ensure unit length
    return normalize(world_normal);
}

/// Combined normal mapping: decode texture and transform to world space
fn sample_normal_map(
    normal_texture: vec4<f32>,
    strength: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    surface_normal: vec3<f32>
) -> vec3<f32> {
    let tangent_normal = decode_normal_map(normal_texture, strength);
    return apply_normal_map(tangent_normal, tangent, bitangent, surface_normal);
}

@fragment  
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample normal map
    let normal_sample = textureSample(normal_texture, normal_sampler, input.uv);
    
    // Decode and apply normal map
    let final_normal = sample_normal_map(
        normal_sample,
        uniforms.normal_strength,
        input.world_tangent,
        input.world_bitangent,
        input.world_normal
    );
    
    // Simple lighting calculation
    let light_dir = normalize(-uniforms.light_direction.xyz);
    let light_intensity = uniforms.light_direction.w;
    
    // Lambertian diffuse lighting
    let ndotl = max(0.0, dot(final_normal, light_dir));
    let diffuse = ndotl * light_intensity;
    
    // Add some ambient lighting
    let ambient = 0.2;
    let lighting = ambient + diffuse;
    
    // Output as grayscale for validation purposes
    let final_color = vec3<f32>(lighting);
    
    return vec4<f32>(final_color, 1.0);
}
