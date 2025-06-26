#version 450
layout(location = 0) in  vec3 inPos;
layout(location = 1) in  vec4 inCol;

layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform MVP
{
    mat4 mvp;
} ubo;

void main()
{
    outColor    = inCol.rgb;
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
