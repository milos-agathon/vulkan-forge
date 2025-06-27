
#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec3 lightDir;
    float time;
} pc;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(pc.lightDir);
    
    // Simple Phong shading
    float ambient = 0.3;
    float diffuse = max(dot(normal, -lightDir), 0.0) * 0.7;
    
    vec3 viewDir = normalize(vec3(0, 0, 1) - fragPos);
    vec3 reflectDir = reflect(lightDir, normal);
    float specular = pow(max(dot(viewDir, reflectDir), 0.0), 32) * 0.5;
    
    vec3 lighting = vec3(ambient + diffuse + specular);
    outColor = vec4(fragColor.rgb * lighting, fragColor.a);
}
