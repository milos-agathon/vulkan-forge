#version 450 core

/*
 * Terrain Vertex Shader
 * 
 * Basic vertex transformation for terrain tessellation pipeline.
 * Passes through vertex data to tessellation control shader.
 */

// Vertex input attributes
layout(location = 0) in vec3 inPosition;     // Base terrain position (typically grid coordinates)
layout(location = 1) in vec2 inTexCoord;    // Texture coordinates for height sampling
layout(location = 2) in vec3 inNormal;      // Base normal (often just up vector)

// Push constants for per-draw parameters
layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;           // Model transformation matrix
    mat4 viewMatrix;            // View matrix
    mat4 projMatrix;            // Projection matrix
    mat4 mvpMatrix;             // Combined MVP matrix for performance
    
    vec3 cameraPosition;        // World space camera position
    float tessellationScale;    // Global tessellation scaling factor
    
    vec2 heightmapSize;         // Size of heightmap texture (width, height)
    vec2 terrainScale;          // Terrain scaling (X, Z)
    float heightScale;          // Height scaling factor
    float time;                 // Elapsed time for animations
    
    // LOD parameters
    float nearDistance;         // Near tessellation distance
    float farDistance;          // Far tessellation distance
    float minTessLevel;         // Minimum tessellation level
    float maxTessLevel;         // Maximum tessellation level
} pc;

// Outputs to tessellation control shader
out VS_OUT {
    vec3 worldPosition;         // World space position
    vec2 texCoord;              // Texture coordinates
    vec3 normal;                // Normal vector
    float distanceToCamera;     // Distance to camera for LOD
} vs_out;

void main() {
    // Calculate world position
    vec4 worldPos = pc.modelMatrix * vec4(inPosition, 1.0);
    vs_out.worldPosition = worldPos.xyz;
    
    // Pass through texture coordinates
    vs_out.texCoord = inTexCoord;
    
    // Transform normal to world space
    mat3 normalMatrix = mat3(transpose(inverse(pc.modelMatrix)));
    vs_out.normal = normalize(normalMatrix * inNormal);
    
    // Calculate distance to camera for LOD decisions
    vs_out.distanceToCamera = length(pc.cameraPosition - worldPos.xyz);
    
    // Pass position to tessellation control shader
    // Note: No projection yet - tessellation happens in world space
    gl_Position = worldPos;
}