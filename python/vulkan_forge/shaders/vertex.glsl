#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec3 fragWorldPos;

// Uniform buffer
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightPositions[4];
    vec4 lightColors[4];
    int numLights;
} ubo;

// Push constants for material
layout(push_constant) uniform Material {
   vec4 baseColor;
   float metallic;
   float roughness;
   float padding1;
   float padding2;
   vec3 emissive;
   float padding3;
} material;

// Output
layout(location = 0) out vec4 outColor;

// Constants
const float PI = 3.14159265359;

// PBR calculations
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
   return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
   float a = roughness * roughness;
   float a2 = a * a;
   float NdotH = max(dot(N, H), 0.0);
   float NdotH2 = NdotH * NdotH;
   
   float num = a2;
   float denom = (NdotH2 * (a2 - 1.0) + 1.0);
   denom = PI * denom * denom;
   
   return num / denom;
}

float geometrySchlickGGX(float NdotV, float roughness) {
   float r = (roughness + 1.0);
   float k = (r * r) / 8.0;
   
   float num = NdotV;
   float denom = NdotV * (1.0 - k) + k;
   
   return num / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
   float NdotV = max(dot(N, V), 0.0);
   float NdotL = max(dot(N, L), 0.0);
   float ggx2 = geometrySchlickGGX(NdotV, roughness);
   float ggx1 = geometrySchlickGGX(NdotL, roughness);
   
   return ggx1 * ggx2;
}

void main() {
   vec3 N = normalize(fragNormal);
   vec3 V = normalize(vec3(inverse(ubo.view)[3]) - fragWorldPos);
   
   vec3 albedo = material.baseColor.rgb;
   float metallic = material.metallic;
   float roughness = material.roughness;
   float ao = 1.0; // Ambient occlusion (could be from texture)
   
   // Calculate reflectance at normal incidence
   vec3 F0 = vec3(0.04);
   F0 = mix(F0, albedo, metallic);
   
   // Reflectance equation
   vec3 Lo = vec3(0.0);
   
   // Calculate contribution from each light
   for (int i = 0; i < ubo.numLights && i < 4; ++i) {
       vec3 lightPos = ubo.lightPositions[i].xyz;
       vec3 lightColor = ubo.lightColors[i].rgb;
       float lightIntensity = ubo.lightColors[i].a;
       
       // Calculate per-light radiance
       vec3 L = normalize(lightPos - fragWorldPos);
       vec3 H = normalize(V + L);
       float distance = length(lightPos - fragWorldPos);
       float attenuation = 1.0 / (distance * distance);
       vec3 radiance = lightColor * lightIntensity * attenuation;
       
       // Cook-Torrance BRDF
       float NDF = distributionGGX(N, H, roughness);
       float G = geometrySmith(N, V, L, roughness);
       vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
       
       vec3 kS = F;
       vec3 kD = vec3(1.0) - kS;
       kD *= 1.0 - metallic;
       
       vec3 numerator = NDF * G * F;
       float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
       vec3 specular = numerator / denominator;
       
       // Add to outgoing radiance Lo
       float NdotL = max(dot(N, L), 0.0);
       Lo += (kD * albedo / PI + specular) * radiance * NdotL;
   }
   
   // Ambient lighting (simple approximation)
   vec3 ambient = vec3(0.03) * albedo * ao;
   vec3 color = ambient + Lo + material.emissive;
   
   // HDR tonemapping
   color = color / (color + vec3(1.0));
   // Gamma correction
   color = pow(color, vec3(1.0/2.2));
   
   outColor = vec4(color, material.baseColor.a);
}

