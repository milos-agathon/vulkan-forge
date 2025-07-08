#version 450 core

/*
 * Terrain Wireframe Fragment Shader
 * 
 * Debug visualization shader for terrain tessellation with:
 * - Wireframe overlay with customizable thickness
 * - Tessellation level color coding
 * - Distance-based line thickness
 * - Multiple visualization modes
 * - Performance-friendly wireframe rendering
 */

// Push constants for wireframe parameters
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
    
    // Wireframe-specific parameters
    vec3 wireframeColor;        // Base wireframe color
    float wireframeThickness;   // Line thickness factor
    float wireframeOpacity;     // Overall opacity
    int visualizationMode;      // 0=wireframe, 1=tessellation, 2=distance, 3=mixed
    
    // Debug colors
    vec3 lowTessColor;          // Color for low tessellation
    vec3 highTessColor;         // Color for high tessellation
    vec3 nearColor;             // Color for near geometry
    vec3 farColor;              // Color for far geometry
} pc;

// Input from tessellation evaluation shader
in TES_OUT {
    vec3 worldPosition;
    vec2 texCoord;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    float distanceToCamera;
    float elevation;
    vec3 viewSpacePosition;
    float lodBlendFactor;
} fs_in;

// Additional wireframe-specific inputs
// These would be computed in geometry shader or via other means
in vec3 barycentric;            // Barycentric coordinates for wireframe
in float tessLevel;             // Actual tessellation level for this fragment

// Output
layout(location = 0) out vec4 fragColor;

// Calculate wireframe intensity using barycentric coordinates
float calculateWireframeIntensity(vec3 bary, float thickness) {
    // Distance to nearest edge
    float minDist = min(min(bary.x, bary.y), bary.z);
    
    // Smooth step for anti-aliased edges
    float edgeIntensity = 1.0 - smoothstep(0.0, thickness, minDist);
    
    return edgeIntensity;
}

// Calculate adaptive wireframe thickness based on distance
float calculateAdaptiveThickness(float distance) {
    // Make lines thicker when farther away to maintain visibility
    float basethickness = pc.wireframeThickness;
    float distanceFactor = 1.0 + (distance / pc.farDistance) * 2.0;
    
    return basethickness * distanceFactor;
}

// Color coding for tessellation levels
vec3 getTessellationLevelColor(float tessLevel) {
    // Normalize tessellation level to 0-1 range
    float normalizedTess = (tessLevel - pc.minTessLevel) / (pc.maxTessLevel - pc.minTessLevel);
    normalizedTess = clamp(normalizedTess, 0.0, 1.0);
    
    // Heat map coloring: blue (low) -> green -> yellow -> red (high)
    vec3 color;
    if (normalizedTess < 0.25) {
        // Blue to cyan
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), normalizedTess * 4.0);
    } else if (normalizedTess < 0.5) {
        // Cyan to green
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (normalizedTess - 0.25) * 4.0);
    } else if (normalizedTess < 0.75) {
        // Green to yellow
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (normalizedTess - 0.5) * 4.0);
    } else {
        // Yellow to red
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (normalizedTess - 0.75) * 4.0);
    }
    
    return color;
}

// Color coding for distance-based LOD
vec3 getDistanceLODColor(float distance) {
    float normalizedDistance = clamp((distance - pc.nearDistance) / (pc.farDistance - pc.nearDistance), 0.0, 1.0);
    
    return mix(pc.nearColor, pc.farColor, normalizedDistance);
}

// Elevation-based color coding
vec3 getElevationColor(float elevation) {
    // Normalize elevation to a reasonable range
    float maxElevation = pc.heightScale; // Assume heightScale is max expected elevation
    float normalizedElevation = clamp(elevation / maxElevation, 0.0, 1.0);
    
    // Purple (low) to white (high) gradient
    return mix(vec3(0.5, 0.0, 0.5), vec3(1.0, 1.0, 1.0), normalizedElevation);
}

// Normal visualization (convert normal to color)
vec3 getNormalColor(vec3 normal) {
    // Convert from [-1,1] to [0,1] range
    return normal * 0.5 + 0.5;
}

// Slope visualization
vec3 getSlopeColor(vec3 normal) {
    float slope = 1.0 - abs(dot(normal, vec3(0.0, 1.0, 0.0)));
    
    // Blue (flat) to red (steep)
    return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), slope);
}

// Grid pattern for texture coordinate visualization
float getGridPattern(vec2 texCoord, float gridSize) {
    vec2 grid = abs(fract(texCoord * gridSize) - 0.5);
    float lineWidth = 0.02;
    
    float gridLines = smoothstep(0.5 - lineWidth, 0.5, max(grid.x, grid.y));
    return gridLines;
}

// Animated effects for dynamic visualization
vec3 addAnimatedEffects(vec3 baseColor, vec2 texCoord) {
    // Pulsing effect based on time
    float pulse = sin(pc.time * 2.0) * 0.5 + 0.5;
    
    // Moving waves across the terrain
    float wave = sin(texCoord.x * 10.0 + pc.time) * sin(texCoord.y * 10.0 + pc.time * 0.7);
    wave = wave * 0.1 + 0.1;
    
    return baseColor + vec3(wave * pulse * 0.3);
}

void main() {
    vec3 finalColor = vec3(0.0);
    float alpha = pc.wireframeOpacity;
    
    // Calculate adaptive wireframe thickness
    float thickness = calculateAdaptiveThickness(fs_in.distanceToCamera);
    
    // Calculate wireframe intensity
    float wireIntensity = calculateWireframeIntensity(barycentric, thickness * 0.01);
    
    // Choose visualization mode
    switch (pc.visualizationMode) {
        case 0: // Pure wireframe
            finalColor = pc.wireframeColor;
            alpha = wireIntensity * pc.wireframeOpacity;
            break;
            
        case 1: // Tessellation level visualization
            {
                vec3 tessColor = getTessellationLevelColor(tessLevel);
                finalColor = mix(tessColor * 0.3, tessColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 2: // Distance-based LOD visualization
            {
                vec3 distColor = getDistanceLODColor(fs_in.distanceToCamera);
                finalColor = mix(distColor * 0.3, distColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 3: // Mixed wireframe + tessellation
            {
                vec3 tessColor = getTessellationLevelColor(tessLevel);
                vec3 wireColor = pc.wireframeColor;
                finalColor = mix(tessColor * 0.5, wireColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 4: // Elevation visualization
            {
                vec3 elevColor = getElevationColor(fs_in.elevation);
                finalColor = mix(elevColor * 0.3, elevColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 5: // Normal visualization
            {
                vec3 normalColor = getNormalColor(fs_in.normal);
                finalColor = mix(normalColor * 0.3, normalColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 6: // Slope visualization
            {
                vec3 slopeColor = getSlopeColor(fs_in.normal);
                finalColor = mix(slopeColor * 0.3, slopeColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 7: // Texture coordinate grid
            {
                float grid = getGridPattern(fs_in.texCoord, 8.0);
                vec3 gridColor = vec3(1.0, 1.0, 0.0);
                finalColor = mix(vec3(0.2), gridColor, grid);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        case 8: // Animated debug mode
            {
                vec3 tessColor = getTessellationLevelColor(tessLevel);
                finalColor = addAnimatedEffects(tessColor, fs_in.texCoord);
                finalColor = mix(finalColor * 0.3, finalColor, wireIntensity);
                alpha = pc.wireframeOpacity;
            }
            break;
            
        default: // Fallback to wireframe
            finalColor = pc.wireframeColor;
            alpha = wireIntensity * pc.wireframeOpacity;
            break;
    }
    
    // Distance-based fading to prevent visual clutter at far distances
    float fadeFactor = 1.0 - clamp((fs_in.distanceToCamera - pc.nearDistance) / pc.farDistance, 0.0, 1.0);
    alpha *= fadeFactor;
    
    // Optional: Add slight depth-based darkening for better depth perception
    float depthFactor = 1.0 - (fs_in.distanceToCamera / pc.farDistance) * 0.3;
    finalColor *= depthFactor;
    
    // Ensure we don't output completely transparent fragments
    alpha = max(alpha, 0.01);
    
    fragColor = vec4(finalColor, alpha);
}