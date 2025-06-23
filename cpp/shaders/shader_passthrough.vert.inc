#version 450 core                           // ★ REQUIRED ★

layout(location = 0) in  vec3 inPos;        // vertex position
layout(location = 1) in  vec2 inUV;         // optional UV

layout(location = 0) out vec2 vUV;          // pass-through to fragment

// trivial MVP uniform (set 0/binding 0)
layout(set = 0, binding = 0) uniform MVP {
    mat4 mvp;
} ubo;

void main()
{
    vUV         = inUV;
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
