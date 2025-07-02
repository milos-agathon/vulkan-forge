#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform MaterialUBO {
    vec4 baseColor;
    float metallic;
    float roughness;
    vec2 padding;
} material;

layout(binding = 2) uniform LightUBO {
    vec4 position[2];
    vec4 color[2];
    vec4 intensity;
} lights;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 color = material.baseColor.rgb * 0.3; // Ambient
    
    for (int i = 0; i < 2; i++) {
        vec3 lightDir = normalize(lights.position[i].xyz - fragWorldPos);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 lightContrib = lights.color[i].rgb * diff * lights.intensity[i];
        color += material.baseColor.rgb * lightContrib * 0.7;
    }
    
    color = clamp(color, 0.0, 1.0);
    outColor = vec4(color, material.baseColor.a);
}