#version 450 core

/*
 * Terrain Fragment Shader
 * 
 * Advanced terrain shading with:
 * - Multi-texture blending based on slope/elevation
 * - Physically-based lighting (PBR)
 * - Distance-based fog
 * - Normal mapping
 * - Procedural detail textures
 * - Atmospheric scattering effects
 */

// Push constants for lighting and rendering parameters
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
    
    // Lighting parameters
    vec3 sunDirection;      // Sun direction (normalized)
    vec3 sunColor;          // Sun color and intensity
    vec3 ambientColor;      // Ambient lighting color
    float shadowIntensity;  // Shadow strength
    
    // Fog parameters  
    vec3 fogColor;          // Fog color
    float fogDensity;       // Fog density
    float fogStart;         // Fog start distance
    float fogEnd;           // Fog end distance
    
    // Material parameters
    float roughness;        // Surface roughness
    float metallic;         // Metallic factor
    float specularPower;    // Specular power for Blinn-Phong
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

// Textures
layout(binding = 0) uniform sampler2D heightTexture;     // Height data
layout(binding = 1) uniform sampler2D normalTexture;    // Pre-computed normals (optional)
layout(binding = 2) uniform sampler2D grassTexture;     // Grass/vegetation texture
layout(binding = 3) uniform sampler2D rockTexture;      // Rock texture
layout(binding = 4) uniform sampler2D sandTexture;      // Sand/dirt texture
layout(binding = 5) uniform sampler2D snowTexture;      // Snow texture
layout(binding = 6) uniform sampler2D detailNormalMap;  // Detail normal map

// Output
layout(location = 0) out vec4 fragColor;

// Constants
const float PI = 3.14159265359;
const vec3 F0_DIELECTRIC = vec3(0.04); // Default F0 for non-metals

// Noise function for procedural details
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Smooth noise
float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // Smoothstep interpolation
    
    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Normal Distribution Function (Trowbridge-Reitz GGX)
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return a2 / denom;
}

// Geometry function (Smith's method)
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float denom = NdotV * (1.0 - k) + k;
    return NdotV / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// Calculate terrain texture blending weights based on slope and elevation
vec4 calculateTextureWeights(vec3 normal, float elevation, float slope) {
    vec4 weights = vec4(0.0);
    
    // Grass: flat areas at low elevation
    weights.x = (1.0 - slope) * (1.0 - clamp(elevation / 100.0, 0.0, 1.0));
    
    // Rock: steep slopes
    weights.y = slope * slope;
    
    // Sand/dirt: moderate slopes and mid elevation
    weights.z = (1.0 - weights.x - weights.y) * clamp(1.0 - elevation / 200.0, 0.0, 1.0);
    
    // Snow: high elevation areas
    weights.w = clamp((elevation - 150.0) / 100.0, 0.0, 1.0);
    
    // Normalize weights
    float totalWeight = weights.x + weights.y + weights.z + weights.w;
    if (totalWeight > 0.0) {
        weights /= totalWeight;
    } else {
        weights.x = 1.0; // Default to grass
    }
    
    return weights;
}

// Sample and blend textures based on weights
vec3 sampleBlendedTexture(vec2 texCoord, vec4 weights, float tileScale) {
    vec2 tiledCoord = texCoord * tileScale;
    
    vec3 grassColor = texture(grassTexture, tiledCoord).rgb;
    vec3 rockColor = texture(rockTexture, tiledCoord).rgb;
    vec3 sandColor = texture(sandTexture, tiledCoord).rgb;
    vec3 snowColor = texture(snowTexture, tiledCoord).rgb;
    
    return grassColor * weights.x + 
           rockColor * weights.y + 
           sandColor * weights.z + 
           snowColor * weights.w;
}

// Calculate fog factor
float calculateFog(float distance) {
    if (distance < pc.fogStart) {
        return 0.0;
    }
    
    float fogRange = pc.fogEnd - pc.fogStart;
    float fogFactor = (distance - pc.fogStart) / fogRange;
    
    // Exponential fog
    fogFactor = 1.0 - exp(-pc.fogDensity * distance * 0.001);
    
    return clamp(fogFactor, 0.0, 1.0);
}

// Simple atmospheric scattering approximation
vec3 calculateAtmosphericScattering(vec3 viewDir, vec3 lightDir) {
    float cosTheta = dot(viewDir, lightDir);
    
    // Rayleigh scattering (blue sky)
    float rayleigh = 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
    
    // Mie scattering (sun disk)
    float g = 0.76; // Anisotropy parameter
    float g2 = g * g;
    float mie = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + cosTheta * cosTheta)) / 
                ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
    
    vec3 skyColor = vec3(0.3, 0.6, 1.0) * rayleigh + vec3(1.0, 0.9, 0.7) * mie;
    
    return skyColor;
}

// Enhanced normal mapping
vec3 calculateDetailNormal(vec2 texCoord, vec3 worldNormal, vec3 tangent, vec3 bitangent) {
    // Sample detail normal map at multiple scales
    vec3 detailNormal1 = texture(detailNormalMap, texCoord * 32.0).rgb * 2.0 - 1.0;
    vec3 detailNormal2 = texture(detailNormalMap, texCoord * 8.0).rgb * 2.0 - 1.0;
    
    // Blend detail normals
    vec3 detailNormal = normalize(detailNormal1 * 0.7 + detailNormal2 * 0.3);
    
    // Transform to world space
    mat3 TBN = mat3(normalize(tangent), normalize(bitangent), normalize(worldNormal));
    vec3 finalNormal = TBN * detailNormal;
    
    return normalize(finalNormal);
}

// Procedural detail enhancement
float calculateProceduralDetail(vec2 worldPos) {
    // Multi-octave noise for surface variation
    float detail = 0.0;
    detail += smoothNoise(worldPos * 0.1) * 0.5;
    detail += smoothNoise(worldPos * 0.2) * 0.25;
    detail += smoothNoise(worldPos * 0.4) * 0.125;
    
    return detail;
}

void main() {
    // Normalize inputs
    vec3 N = normalize(fs_in.normal);
    vec3 V = normalize(pc.cameraPosition - fs_in.worldPosition);
    vec3 L = normalize(-pc.sunDirection);
    vec3 H = normalize(V + L);
    
    // Calculate slope for texture blending
    float slope = 1.0 - abs(dot(N, vec3(0.0, 1.0, 0.0)));
    
    // Enhanced normal with detail mapping
    vec3 detailNormal = calculateDetailNormal(fs_in.texCoord, N, fs_in.tangent, fs_in.bitangent);
    N = mix(N, detailNormal, clamp(1.0 - fs_in.lodBlendFactor, 0.0, 1.0));
    
    // Recalculate lighting vectors with enhanced normal
    H = normalize(V + L);
    
    // Calculate texture blending weights
    vec4 textureWeights = calculateTextureWeights(N, fs_in.elevation, slope);
    
    // Sample base color from blended textures
    vec3 albedo = sampleBlendedTexture(fs_in.texCoord, textureWeights, 16.0);
    
    // Add procedural variation
    float proceduralDetail = calculateProceduralDetail(fs_in.worldPosition.xz);
    albedo = mix(albedo, albedo * 1.2, proceduralDetail * 0.1);
    
    // Material properties
    float roughness = pc.roughness;
    float metallic = pc.metallic;
    
    // Adjust material properties based on texture weights
    roughness = mix(roughness, 0.9, textureWeights.y); // Rock is rougher
    roughness = mix(roughness, 0.1, textureWeights.w); // Snow is smoother
    metallic = mix(metallic, 0.0, textureWeights.x + textureWeights.z + textureWeights.w); // Only rock has some metallic
    
    // Calculate F0 for PBR
    vec3 F0 = mix(F0_DIELECTRIC, albedo, metallic);
    
    // Lighting calculations
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    
    // PBR BRDF
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001; // Prevent division by zero
    vec3 specular = numerator / denominator;
    
    // Energy conservation
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    // Lambertian diffuse
    vec3 diffuse = kD * albedo / PI;
    
    // Direct lighting
    vec3 radiance = pc.sunColor * NdotL;
    vec3 color = (diffuse + specular) * radiance;
    
    // Ambient lighting with simple hemisphere lighting
    vec3 ambientUp = pc.ambientColor * 1.2;
    vec3 ambientDown = pc.ambientColor * 0.4;
    float hemisphereBlend = dot(N, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    vec3 ambient = mix(ambientDown, ambientUp, hemisphereBlend) * albedo;
    color += ambient;
    
    // Simple subsurface scattering for vegetation
    if (textureWeights.x > 0.5) { // Grass areas
        float backScatter = clamp(dot(-L, V), 0.0, 1.0);
        vec3 subsurface = albedo * pc.sunColor * backScatter * 0.3;
        color += subsurface;
    }
    
    // Atmospheric scattering
    vec3 atmosphericColor = calculateAtmosphericScattering(-V, L);
    color += atmosphericColor * 0.1;
    
    // Distance-based fog
    float fogFactor = calculateFog(fs_in.distanceToCamera);
    color = mix(color, pc.fogColor, fogFactor);
    
    // Tone mapping (simple Reinhard)
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec2(1.0/2.2));
    
    // Debug: Visualize tessellation levels (uncomment for debugging)
    // float tessDebug = fs_in.lodBlendFactor;
    // color = mix(color, vec3(tessDebug, 0.0, 1.0 - tessDebug), 0.3);
    
    fragColor = vec4(color, 1.0);
}