// File: python/vulkan_forge/shaders/fragment.glsl
#version 450

//---------------------------------------------
// Inputs from vertex shader
//---------------------------------------------
layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vWorldPos;

//---------------------------------------------
// Output
//---------------------------------------------
layout(location = 0) out vec4 outColor;

//---------------------------------------------
// Material push-constants (std430 for tight packing)
// metallic  = mr.x
// roughness = mr.y
//---------------------------------------------
layout(push_constant, std430) uniform PushConstants {
    vec4 baseColor;   // rgba
    vec2 mr;          // x = metallic, y = roughness
} pc;

//---------------------------------------------
// Single, hard-coded directional light
//---------------------------------------------
const vec3 kLightDir = normalize(vec3(1.0, 1.0, 1.0));
const vec3 kAmbient  = vec3(0.3);

void main()
{
    vec3  N      = normalize(vNormal);
    float NdotL  = max(dot(N, kLightDir), 0.0);

    // Simple Lambert diffuse + ambient
    vec3 diffuse = pc.baseColor.rgb * NdotL;
    vec3 color   = diffuse + pc.baseColor.rgb * kAmbient;

    // (optional) apply gamma if swap-chain is UNORM; skip if SRGB
    // color = pow(color, vec3(1.0/2.2));

    outColor = vec4(color, pc.baseColor.a);
}
