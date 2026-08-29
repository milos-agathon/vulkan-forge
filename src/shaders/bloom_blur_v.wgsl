// Q5: Bloom vertical blur pass
// Applies vertical Gaussian blur for bloom effect

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> uniforms: BloomBlurUniforms;

struct BloomBlurUniforms {
    radius: f32,
    strength: f32,
    _pad: vec2<f32>,
}

// Gaussian weights for 9-tap blur kernel
// Pre-calculated weights for sigma = 2.0
var<private> WEIGHTS: array<f32, 9> = array<f32, 9>(
    0.0077847, 0.0231017, 0.0539909, 0.0995906, 0.1420118,
    0.1599471, 0.1420118, 0.0995906, 0.0539909
);

var<private> OFFSETS: array<i32, 9> = array<i32, 9>(-4, -3, -2, -1, 0, 1, 2, 3, 4);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coord = global_id.xy;
    
    if coord.x >= dimensions.x || coord.y >= dimensions.y {
        return;
    }
    
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    // Apply vertical blur
    for (var i = 0; i < 9; i++) {
        let offset_y = i32(coord.y) + i32(f32(OFFSETS[i]) * uniforms.radius);
        let sample_coord = vec2<i32>(i32(coord.x), offset_y);
        
        // Clamp coordinates to texture bounds
        if sample_coord.y >= 0 && sample_coord.y < i32(dimensions.y) {
            let sample_color = textureLoad(input_texture, vec2<u32>(sample_coord), 0);
            let weight = WEIGHTS[i];
            
            color += sample_color * weight;
            total_weight += weight;
        }
    }
    
    // Normalize and apply strength
    if total_weight > 0.0 {
        color = color / total_weight;
    }
    
    // Apply blur strength (blend with original)
    let original = textureLoad(input_texture, coord, 0);
    let result = mix(original, color, uniforms.strength);
    
    textureStore(output_texture, coord, result);
}
