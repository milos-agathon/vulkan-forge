#version 450 core

/*
 * Terrain Tessellation Control Shader
 * 
 * Determines tessellation levels based on:
 * - Distance to camera (LOD)
 * - Screen-space projected size
 * - Terrain curvature/slope
 * - Performance budgets
 */

// Control how many vertices per patch (quad patches = 4)
layout(vertices = 4) out;

// Push constants (same as vertex shader)
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

// Input from vertex shader
in VS_OUT {
    vec3 worldPosition;
    vec2 texCoord;
    vec3 normal;
    float distanceToCamera;
} vs_in[];

// Output to tessellation evaluation shader
out TCS_OUT {
    vec3 worldPosition;
    vec2 texCoord;
    vec3 normal;
    float distanceToCamera;
} tcs_out[];

// Height sampling for curvature analysis
layout(binding = 0) uniform sampler2D heightTexture;

// Calculate tessellation level based on distance
float calculateDistanceTessLevel(float distance) {
    // Logarithmic falloff for smooth LOD transitions
    float normalizedDistance = clamp((distance - pc.nearDistance) / (pc.farDistance - pc.nearDistance), 0.0, 1.0);
    
    // Exponential falloff for more aggressive LOD
    float factor = 1.0 - normalizedDistance * normalizedDistance;
    
    return mix(pc.minTessLevel, pc.maxTessLevel, factor);
}

// Calculate tessellation level based on screen-space projection
float calculateScreenSpaceTessLevel(vec3 worldPos0, vec3 worldPos1) {
    // Project edge endpoints to screen space
    vec4 clip0 = pc.mvpMatrix * vec4(worldPos0, 1.0);
    vec4 clip1 = pc.mvpMatrix * vec4(worldPos1, 1.0);
    
    // Convert to NDC
    vec2 ndc0 = clip0.xy / clip0.w;
    vec2 ndc1 = clip1.xy / clip1.w;
    
    // Calculate screen-space edge length
    vec2 screenSize = vec2(1920.0, 1080.0); // TODO: Pass as uniform
    vec2 screen0 = (ndc0 * 0.5 + 0.5) * screenSize;
    vec2 screen1 = (ndc1 * 0.5 + 0.5) * screenSize;
    
    float screenLength = length(screen1 - screen0);
    
    // Target 4-8 pixels per tessellated edge
    float targetPixelsPerEdge = 6.0;
    float tessLevel = screenLength / targetPixelsPerEdge;
    
    return clamp(tessLevel, pc.minTessLevel, pc.maxTessLevel);
}

// Sample height at texture coordinates
float sampleHeight(vec2 texCoord) {
    // Clamp texture coordinates to valid range
    vec2 clampedCoord = clamp(texCoord, vec2(0.0), vec2(1.0));
    return texture(heightTexture, clampedCoord).r * pc.heightScale;
}

// Calculate tessellation level based on terrain curvature
float calculateCurvatureTessLevel(vec2 centerTexCoord) {
    // Sample heights around center point
    float texelSize = 1.0 / max(pc.heightmapSize.x, pc.heightmapSize.y);
    
    float centerHeight = sampleHeight(centerTexCoord);
    float leftHeight = sampleHeight(centerTexCoord + vec2(-texelSize, 0.0));
    float rightHeight = sampleHeight(centerTexCoord + vec2(texelSize, 0.0));
    float upHeight = sampleHeight(centerTexCoord + vec2(0.0, texelSize));
    float downHeight = sampleHeight(centerTexCoord + vec2(0.0, -texelSize));
    
    // Calculate curvature approximation
    float curvatureX = abs(leftHeight + rightHeight - 2.0 * centerHeight);
    float curvatureY = abs(upHeight + downHeight - 2.0 * centerHeight);
    float maxCurvature = max(curvatureX, curvatureY);
    
    // Scale curvature to tessellation level
    float curvatureScale = 10.0; // Adjust based on terrain characteristics
    float curvatureTessLevel = maxCurvature * curvatureScale;
    
    return clamp(curvatureTessLevel, 0.0, pc.maxTessLevel - pc.minTessLevel);
}

void main() {
    // Pass through vertex data
    tcs_out[gl_InvocationID].worldPosition = vs_in[gl_InvocationID].worldPosition;
    tcs_out[gl_InvocationID].texCoord = vs_in[gl_InvocationID].texCoord;
    tcs_out[gl_InvocationID].normal = vs_in[gl_InvocationID].normal;
    tcs_out[gl_InvocationID].distanceToCamera = vs_in[gl_InvocationID].distanceToCamera;
    
    // Pass through position for tessellation evaluation
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    
    // Only the first invocation calculates tessellation levels
    if (gl_InvocationID == 0) {
        // Get patch corners
        vec3 pos0 = vs_in[0].worldPosition;
        vec3 pos1 = vs_in[1].worldPosition;
        vec3 pos2 = vs_in[2].worldPosition;
        vec3 pos3 = vs_in[3].worldPosition;
        
        vec2 tex0 = vs_in[0].texCoord;
        vec2 tex1 = vs_in[1].texCoord;
        vec2 tex2 = vs_in[2].texCoord;
        vec2 tex3 = vs_in[3].texCoord;
        
        // Calculate center position for distance-based LOD
        vec3 centerPos = (pos0 + pos1 + pos2 + pos3) * 0.25;
        vec2 centerTex = (tex0 + tex1 + tex2 + tex3) * 0.25;
        float centerDistance = length(pc.cameraPosition - centerPos);
        
        // Calculate base tessellation level from distance
        float baseTessLevel = calculateDistanceTessLevel(centerDistance);
        
        // Apply global tessellation scale
        baseTessLevel *= pc.tessellationScale;
        
        // Calculate edge-specific tessellation levels
        float tessLevel0 = baseTessLevel; // Bottom edge (0->1)
        float tessLevel1 = baseTessLevel; // Right edge (1->2)  
        float tessLevel2 = baseTessLevel; // Top edge (2->3)
        float tessLevel3 = baseTessLevel; // Left edge (3->0)
        
        // Enhance with screen-space considerations
        if (centerDistance < pc.farDistance * 0.5) {
            tessLevel0 = max(tessLevel0, calculateScreenSpaceTessLevel(pos0, pos1));
            tessLevel1 = max(tessLevel1, calculateScreenSpaceTessLevel(pos1, pos2));
            tessLevel2 = max(tessLevel2, calculateScreenSpaceTessLevel(pos2, pos3));
            tessLevel3 = max(tessLevel3, calculateScreenSpaceTessLevel(pos3, pos0));
        }
        
        // Enhance with curvature for nearby patches
        if (centerDistance < pc.nearDistance * 2.0) {
            float curvatureBoost = calculateCurvatureTessLevel(centerTex);
            tessLevel0 += curvatureBoost;
            tessLevel1 += curvatureBoost;
            tessLevel2 += curvatureBoost;
            tessLevel3 += curvatureBoost;
        }
        
        // Clamp all levels to valid range
        tessLevel0 = clamp(tessLevel0, pc.minTessLevel, pc.maxTessLevel);
        tessLevel1 = clamp(tessLevel1, pc.minTessLevel, pc.maxTessLevel);
        tessLevel2 = clamp(tessLevel2, pc.minTessLevel, pc.maxTessLevel);
        tessLevel3 = clamp(tessLevel3, pc.minTessLevel, pc.maxTessLevel);
        
        // Calculate inner tessellation levels
        // Use average of opposite edges for smooth transitions
        float innerTessLevel0 = (tessLevel0 + tessLevel2) * 0.5; // U direction
        float innerTessLevel1 = (tessLevel1 + tessLevel3) * 0.5; // V direction
        
        // Set tessellation levels
        gl_TessLevelOuter[0] = tessLevel0; // Bottom edge
        gl_TessLevelOuter[1] = tessLevel1; // Right edge  
        gl_TessLevelOuter[2] = tessLevel2; // Top edge
        gl_TessLevelOuter[3] = tessLevel3; // Left edge
        
        gl_TessLevelInner[0] = innerTessLevel0; // U direction
        gl_TessLevelInner[1] = innerTessLevel1; // V direction
        
        // Debug: Uncomment to visualize tessellation levels
        // if (centerDistance < 100.0) {
        //     gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = 
        //     gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = pc.maxTessLevel;
        //     gl_TessLevelInner[0] = gl_TessLevelInner[1] = pc.maxTessLevel;
        // }
    }
}