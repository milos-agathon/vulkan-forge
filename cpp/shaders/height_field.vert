
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec4 fragColor;
layout(location = 2) out vec3 fragPos;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec3 lightDir;
    float time;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragNormal = inNormal;
    fragColor = inColor;
    fragPos = inPosition;
}
