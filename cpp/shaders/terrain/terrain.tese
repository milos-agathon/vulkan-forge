#version 450 core

/*
 * Terrain Tessellation Evaluation Shader
 * 
 * Generates final vertex positions by:
 * - Interpolating base terrain coordinates
 * - Sampling heightmap for elevation
 * - Computing accurate normals
 * - Applying morphing between LOD levels
 */

// Tessellation primitive: quads with equal spacing
layout(quads, equal_spacing, ccw) in;

// Push constants (same as previous shaders)
layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projMatrix;
    mat4 mvpMatrix;
    
    vec3 cameraPosition;
    float tessellationScale;
    
    vec2 heightmapSize;
    vec2 terrainScale;
    float heightScale;
    float time;
    
    float nearDistance;
    float farDistance;
    float minTessLevel;
    float maxTessLevel;
} pc;

// Input from tessellation control shader
in TCS_OUT {
    vec3 worldPosition;
    vec2 texCoord;
    vec3 normal;
    float distanceToCamera;
} tes_in[];

// Output to fragment shader
out TES_OUT {
    vec3 worldPosition;      // Final world position including height
    vec2 texCoord;           // Texture coordinates for fragment shader
    vec3 normal;             // Calculated surface normal
    vec3 tangent;            // Tangent vector for normal mapping
    vec3 bitangent;          // Bitangent vector for normal mapping
    float distanceToCamera;  // Distance for fog/LOD effects
    float elevation;         // Raw elevation value
    vec3 viewSpacePosition;  // Position in view space
    float lodBlendFactor;    // Factor for morphing between LOD levels
} tes_out;

// Textures
layout(binding = 0) uniform sampler2D heightTexture;    // Primary heightmap
layout(binding = 1) uniform sampler2D normalTexture;    // Optional pre-computed normals

// Bilinear interpolation of patch data
vec3 interpolateVec3(vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec2 uv) {
    vec3 bottom = mix(v0, v1, uv.x);
    vec3 top = mix(v3, v2, uv.x);
    return mix(bottom, top, uv.y);
}

vec2 interpolateVec2(vec2 v0, vec2 v1, vec2 v2, vec2 v3, vec2 uv) {
    vec2 bottom = mix(v0, v1, uv.x);
    vec2 top = mix(v3, v2, uv.x);
    return mix(bottom, top, uv.y);
}

float interpolateFloat(float f0, float f1, float f2, float f3, vec2 uv) {
    float bottom = mix(f0, f1, uv.x);
    float top = mix(f3, f2, uv.x);
    return mix(bottom, top, uv.y);
}

// High-quality height sampling with filtering
float sampleHeight(vec2 texCoord) {
    // Clamp coordinates to prevent edge artifacts
    vec2 clampedCoord = clamp(texCoord, vec2(0.0), vec2(1.0));
    
    // Sample height with bilinear filtering
    return texture(heightTexture, clampedCoord).r * pc.heightScale;
}

// Calculate surface normal using finite differences
vec3 calculateNormal(vec2 texCoord, float heightScale) {
    float texelSize = 1.0 / max(pc.heightmapSize.x, pc.heightmapSize.y);
    
    // Sample neighboring heights
    float hL = sampleHeight(texCoord + vec2(-texelSize, 0.0)); // Left
    float hR = sampleHeight(texCoord + vec2(texelSize, 0.0));  // Right
    float hD = sampleHeight(texCoord + vec2(0.0, -texelSize)); // Down
    float hU = sampleHeight(texCoord + vec2(0.0, texelSize));  // Up
    
    // Calculate normal using cross product of tangent vectors
    vec3 tangentX = vec3(2.0 * texelSize * pc.terrainScale.x, hR - hL, 0.0);
    vec3 tangentZ = vec3(0.0, hU - hD, 2.0 * texelSize * pc.terrainScale.y);
    
    return normalize(cross(tangentZ, tangentX));
}

// Calculate tangent and bitangent vectors for normal mapping
void calculateTangentSpace(vec2 texCoord, out vec3 tangent, out vec3 bitangent, vec3 normal) {
    float texelSize = 1.0 / max(pc.heightmapSize.x, pc.heightmapSize.y);
    
    // Calculate height gradients
    float hL = sampleHeight(texCoord + vec2(-texelSize, 0.0));
    float hR = sampleHeight(texCoord + vec2(texelSize, 0.0));
    float hD = sampleHeight(texCoord + vec2(0.0, -texelSize));
    float hU = sampleHeight(texCoord + vec2(0.0, texelSize));
    
    // Tangent along X (U) direction
    tangent = normalize(vec3(2.0 * texelSize * pc.terrainScale.x, hR - hL, 0.0));
    
    // Bitangent along Z (V) direction  
    bitangent = normalize(vec3(0.0, hU - hD, 2.0 * texelSize * pc.terrainScale.y));
    
    // Orthogonalize using Gram-Schmidt process
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    bitangent = normalize(cross(normal, tangent));
}

// LOD morphing to smooth transitions between tessellation levels
vec3 applyLODMorphing(vec3 position, vec2 texCoord, float distance) {
    // Morph factor based on distance
    float morphRange = (pc.farDistance - pc.nearDistance) * 0.3;
    float morphStart = pc.nearDistance + morphRange;
    float morphFactor = clamp((distance - morphStart) / morphRange, 0.0, 1.0);
    
    if (morphFactor > 0.0) {
        // Sample lower LOD height (simulate lower tessellation)
        vec2 lowerLODCoord = floor(texCoord * pc.heightmapSize * 0.5) / (pc.heightmapSize * 0.5);
        float lowerLODHeight = sampleHeight(lowerLODCoord);
        
        // Blend between current and lower LOD
        position.y = mix(position.y, lowerLODHeight, morphFactor * 0.5);
    }
    
    return position;
}

void main() {
    // Get tessellation coordinates (0 to 1 in both U and V)
    vec2 tessCoord = gl_TessCoord.xy;
    
    // Interpolate position data from patch corners
    vec3 worldPos = interpolateVec3(
        tes_in[0].worldPosition,
        tes_in[1].worldPosition, 
        tes_in[2].worldPosition,
        tes_in[3].worldPosition,
        tessCoord
    );
    
    // Interpolate texture coordinates
    vec2 texCoord = interpolateVec2(
        tes_in[0].texCoord,
        tes_in[1].texCoord,
        tes_in[2].texCoord, 
        tes_in[3].texCoord,
        tessCoord
    );
    
    // Interpolate distance for smooth LOD transitions
    float distance = interpolateFloat(
        tes_in[0].distanceToCamera,
        tes_in[1].distanceToCamera,
        tes_in[2].distanceToCamera,
        tes_in[3].distanceToCamera,
        tessCoord
    );
    
    // Sample height from heightmap
    float height = sampleHeight(texCoord);
    
    // Apply height to world position
    worldPos.y = height;
    
    // Apply LOD morphing for smooth transitions
    worldPos = applyLODMorphing(worldPos, texCoord, distance);
    
    // Calculate surface normal
    vec3 normal = calculateNormal(texCoord, pc.heightScale);
    
    // Calculate tangent space for normal mapping
    vec3 tangent, bitangent;
    calculateTangentSpace(texCoord, tangent, bitangent, normal);
    
    // Transform to view space for fragment shader
    vec4 viewSpacePos = pc.viewMatrix * vec4(worldPos, 1.0);
    
    // Calculate LOD blend factor for fragment shader
    float lodBlendFactor = clamp((distance - pc.nearDistance) / (pc.farDistance - pc.nearDistance), 0.0, 1.0);
    
    // Output to fragment shader
    tes_out.worldPosition = worldPos;
    tes_out.texCoord = texCoord;
    tes_out.normal = normal;
    tes_out.tangent = tangent;
    tes_out.bitangent = bitangent;
    tes_out.distanceToCamera = distance;
    tes_out.elevation = height;
    tes_out.viewSpacePosition = viewSpacePos.xyz;
    tes_out.lodBlendFactor = lodBlendFactor;
    
    // Final vertex position in clip space
    gl_Position = pc.projMatrix * viewSpacePos;
}