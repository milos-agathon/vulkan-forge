#version 450
layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 vPos;
void main() {
    vPos = inPos;
    gl_Position = vec4(inPos, 1.0);
}
