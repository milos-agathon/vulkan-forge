#version 450 core

layout(location = 0) in  vec3 inColor;  // from your vertex shader
layout(location = 0) out vec4 outFrag;   // to the framebuffer

void main()
{
    outFrag = vec4(inColor, 1.0);
}
