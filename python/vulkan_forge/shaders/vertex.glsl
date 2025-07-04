// File: python/vulkan_forge/shaders/vertex.glsl
#version 450

//---------------------------------------------
// Vertex attributes
//---------------------------------------------
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

//---------------------------------------------
// Per-draw uniform block  (set = 0, binding = 0)
// NOTE: explicit *set* qualifier avoids descriptor-set mismatches
//---------------------------------------------
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;          // supply a Vulkan-style projection (Y already flipped) on the CPU
} ubo;

//---------------------------------------------
// Outputs to fragment shader
//---------------------------------------------
layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorldPos;

void main()
{
    // World-space position
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    vWorldPos     = worldPos.xyz;

    // Normal matrix (per-vertex inverse-transpose is OK for bootstrap; optimise later)
    vNormal = mat3(transpose(inverse(ubo.model))) * inNormal;

    vUV = inUV;

    // Final clip-space position (projection should already be Vulkan-correct)
    gl_Position = ubo.proj * ubo.view * worldPos;
}
