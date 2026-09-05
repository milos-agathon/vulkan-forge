// src/shaders/hybrid_kernel.wgsl
// Main compute kernel for hybrid path tracing combining SDF raymarching with BVH traversal
// Extends pt_kernel.wgsl functionality with SDF support

#include "hybrid_traversal.wgsl"

// Base uniforms (Group 0)
struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  aov_flags: u32,
  cam_origin: vec3<f32>,
  cam_fov_y: f32,
  cam_right: vec3<f32>,
  cam_aspect: f32,
  cam_up: vec3<f32>,
  cam_exposure: f32,
  cam_forward: vec3<f32>,
  seed_hi: u32,
  seed_lo: u32,
  camera_model: u32,
  full_width: u32,
  full_height: u32,
  pixel_offset_x: u32,
  pixel_offset_y: u32,
  ortho_half_height: f32,
  camera_flags: u32,
  sensor_rect: vec4<f32>,
}

// Lighting uniforms (Group 0, binding 1 — folded into the base uniform group
// so the whole hybrid pipeline fits within max_bind_groups = 4)
struct LightingUniforms {
  light_dir: vec3<f32>,           // Directional light direction
  lighting_type: u32,             // 0=flat, 1=lambertian, 2=phong, 3=blinn-phong
  light_color: vec3<f32>,         // Light color * intensity
  shadows_enabled: u32,           // 0 or 1
  ambient_color: vec3<f32>,       // Ambient/indirect light
  shadow_intensity: f32,          // Shadow darkness [0,1]
  hdri_intensity: f32,            // HDR environment intensity
  hdri_rotation: f32,             // HDR rotation in radians
  specular_power: f32,            // Phong/Blinn specular exponent
  _pad: vec3<u32>,
}

// Sphere primitive for legacy support
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    _pad0: f32
}

// Bind groups (0-3 only: portable to max_bind_groups = 4 adapters)
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> lighting: LightingUniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(2) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;
@group(3) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;

// AOV Output textures (moved into Group 3 to stay within max_bind_groups)
@group(3) @binding(1) var aov_albedo: texture_storage_2d<rgba16float, write>;
@group(3) @binding(2) var aov_normal: texture_storage_2d<rgba16float, write>;
@group(3) @binding(3) var aov_depth: texture_storage_2d<r32float, write>;
@group(3) @binding(4) var aov_direct: texture_storage_2d<rgba16float, write>;
@group(3) @binding(5) var aov_indirect: texture_storage_2d<rgba16float, write>;
@group(3) @binding(6) var aov_emission: texture_storage_2d<rgba16float, write>;
@group(3) @binding(7) var aov_visibility: texture_storage_2d<rgba8unorm, write>;

// AOV flag constants
const AOV_ALBEDO_BIT: u32 = 0u;
const AOV_NORMAL_BIT: u32 = 1u;
const AOV_DEPTH_BIT: u32 = 2u;
const AOV_DIRECT_BIT: u32 = 3u;
const AOV_INDIRECT_BIT: u32 = 4u;
const AOV_EMISSION_BIT: u32 = 5u;
const AOV_VISIBILITY_BIT: u32 = 6u;

fn aov_enabled(bit: u32) -> bool {
    return (uniforms.aov_flags & (1u << bit)) != 0u;
}

// Random number generation
fn xorshift32(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= (x << 13u);
  x ^= (x >> 17u);
  x ^= (x << 5u);
  *state = x;
  return f32(x) / 4294967296.0;
}

// Tent filter for antialiasing
fn tent_filter(u: f32) -> f32 {
  let t = 2.0 * u - 1.0;
  return select(1.0 + t, 1.0 - t, t < 0.0);
}

struct CameraRay {
  origin: vec3<f32>,
  direction: vec3<f32>,
}

fn global_pixel(gid: vec2<u32>) -> vec2<u32> {
  return gid + vec2<u32>(uniforms.pixel_offset_x, uniforms.pixel_offset_y);
}

// Seamless calls map global pixel IDs directly onto the ideal full sensor.
// The host validates sensor_rect against this pixel offset and tile extent.
fn generate_camera_ray(gid: vec2<u32>, jitter: vec2<f32>) -> CameraRay {
  if (uniforms.camera_flags == 0u) {
    // Preserve the original arithmetic path byte-for-byte for legacy callers.
    let ndc = vec2<f32>(
      ((f32(gid.x) + 0.5 + jitter.x) / f32(uniforms.width)) * 2.0 - 1.0,
      (1.0 - (f32(gid.y) + 0.5 + jitter.y) / f32(uniforms.height)) * 2.0 - 1.0,
    );
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let rd = normalize(vec3<f32>(ndc.x * half_w, ndc.y * half_h, -1.0));
    return CameraRay(
      uniforms.cam_origin,
      normalize(
        rd.x * uniforms.cam_right + rd.y * uniforms.cam_up
        + rd.z * (-uniforms.cam_forward)
      ),
    );
  }

  let validated_sensor_rect = uniforms.sensor_rect;
  let gpx = global_pixel(gid);
  let sensor_uv = (vec2<f32>(gpx) + vec2<f32>(0.5) + jitter)
      / vec2<f32>(f32(uniforms.full_width), f32(uniforms.full_height));
  let ndc = vec2<f32>(sensor_uv.x * 2.0 - 1.0, (1.0 - sensor_uv.y) * 2.0 - 1.0);
  let aspect = f32(uniforms.full_width) / f32(uniforms.full_height);
  if (uniforms.camera_model == 1u) {
    let half_h = uniforms.ortho_half_height;
    let half_w = aspect * half_h;
    return CameraRay(
      uniforms.cam_origin
          + ndc.x * half_w * uniforms.cam_right
          + ndc.y * half_h * uniforms.cam_up,
      normalize(uniforms.cam_forward),
    );
  }
  let half_h = tan(0.5 * uniforms.cam_fov_y);
  let half_w = aspect * half_h;
  let rd = normalize(vec3<f32>(ndc.x * half_w, ndc.y * half_h, -1.0));
  return CameraRay(
    uniforms.cam_origin,
    normalize(
        rd.x * uniforms.cam_right + rd.y * uniforms.cam_up
        + rd.z * (-uniforms.cam_forward)
    ),
  );
}

// Ray-sphere intersection (for legacy sphere support)
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let oc = ro - c;
  let b = dot(oc, rd);
  let cterm = dot(oc, oc) - r * r;
  let disc = b * b - cterm;
  if (disc <= 0.0) { return 1e30; }
  let s = sqrt(disc);
  let t0 = -b - s;
  let t1 = -b + s;
  if (t0 > 1e-3) { return t0; }
  if (t1 > 1e-3) { return t1; }
  return 1e30;
}

// Simple tonemap function
fn reinhard_tonemap(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = color * exposure;
    return exposed / (vec3<f32>(1.0) + exposed);
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = uniforms.width;
  let H = uniforms.height;
  if (gid.x >= W || gid.y >= H) { return; }
  let px = f32(gid.x);
  let py = f32(gid.y);

  // Seed per-pixel RNG
  var st: u32 = uniforms.seed_hi ^ (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ uniforms.frame_index;
  let jx = tent_filter(xorshift32(&st)) * 0.5;
  let jy = tent_filter(xorshift32(&st)) * 0.5;

  // Generate camera ray
  let ndc_x = ((px + 0.5 + jx) / f32(W)) * 2.0 - 1.0;
  let ndc_y = (1.0 - (py + 0.5 + jy) / f32(H)) * 2.0 - 1.0;
  let half_h = tan(0.5 * uniforms.cam_fov_y);
  let half_w = uniforms.cam_aspect * half_h;
  var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
  rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
  let ro = uniforms.cam_origin;

  // Create ray
  let camera_ray = Ray(ro, 1e-3, rd, 1e30);

  // Initialize results
  var t_best = 1e30;
  var hit_albedo = vec3<f32>(0.7, 0.7, 0.8);
  var hit_normal = vec3<f32>(0.0, 0.0, 1.0);
  var hit_material_type = 0u; // 0 = mesh, 1 = sphere, 2 = SDF
  var hit_point = vec3<f32>(0.0);

  // Test legacy spheres first
  let sphere_count = arrayLength(&scene_spheres);
  for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
    let s = scene_spheres[i];
    let t = ray_sphere(ro, rd, s.center, s.radius);
    if (t < t_best) {
      t_best = t;
      hit_point = ro + rd * t;
      hit_normal = normalize(hit_point - s.center);
      hit_albedo = s.albedo;
      hit_material_type = 1u;
    }
  }

  // Test hybrid scene (SDF + mesh)
  let hybrid_hit = intersect_hybrid(camera_ray);
  if (hybrid_hit.hit != 0u && hybrid_hit.t < t_best) {
    t_best = hybrid_hit.t;
    hit_point = hybrid_hit.point;
    hit_normal = hybrid_hit.normal;
    hit_albedo = get_surface_properties(hybrid_hit);
    hit_material_type = hybrid_hit.hit_type + 2u; // 2 = mesh, 3 = SDF
  }

  // Calculate final color
  let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let is_hit = t_best < 1e20;

  // Sky color for miss cases - use magenta marker for easy detection
  var sky_color = vec3<f32>(0.0);
  if (!is_hit) {
    // Magenta (1.0, 0.0, 1.0) - won't occur naturally, easy to detect in Python
    sky_color = vec3<f32>(1.0, 0.0, 1.0);
  }

  // Lighting
  var diffuse_light = vec3<f32>(0.0);
  var specular_light = vec3<f32>(0.0);
  var indirect_light = lighting.ambient_color;
  var final_color = sky_color;

  if (is_hit) {
    let view_dir = normalize(ro - hit_point);
    
    // Calculate shadow factor
    var shadow_factor = 1.0;
    if (lighting.shadows_enabled != 0u) {
      let shadow_ray = Ray(hit_point + hit_normal * 0.001, 0.001, lighting.light_dir, 1000.0);
      let raw_shadow = soft_shadow_factor(shadow_ray, 1000.0, 4.0);
      // Apply shadow_intensity: 0.0=full shadow, 1.0=no shadow
      shadow_factor = mix(raw_shadow, 1.0, 1.0 - lighting.shadow_intensity);
    }

    // Compute lighting based on lighting_type
    // 0=flat, 1=lambertian, 2=phong, 3=blinn-phong
    
    if (lighting.lighting_type == 0u) {
      // Flat shading - uniform lighting, no direction
      diffuse_light = lighting.light_color * shadow_factor;
      specular_light = vec3<f32>(0.0);
    } else if (lighting.lighting_type == 1u) {
      // Lambertian (diffuse only, no specular)
      let ndotl = max(0.0, dot(hit_normal, lighting.light_dir));
      diffuse_light = lighting.light_color * ndotl * shadow_factor;
      specular_light = vec3<f32>(0.0);
    } else if (lighting.lighting_type == 2u) {
      // Phong (diffuse + sharp specular highlight)
      let ndotl = max(0.0, dot(hit_normal, lighting.light_dir));
      diffuse_light = lighting.light_color * ndotl * shadow_factor;
      
      let reflect_dir = reflect(-lighting.light_dir, hit_normal);
      let spec_intensity = pow(max(0.0, dot(view_dir, reflect_dir)), lighting.specular_power);
      // Specular is NOT shadowed and NOT multiplied by albedo (it's direct reflection)
      specular_light = lighting.light_color * spec_intensity * 0.2;
    } else if (lighting.lighting_type == 3u) {
      // Blinn-Phong (diffuse + softer specular with halfway vector)
      let ndotl = max(0.0, dot(hit_normal, lighting.light_dir));
      diffuse_light = lighting.light_color * ndotl * shadow_factor;
      
      let halfway = normalize(lighting.light_dir + view_dir);
      // Blinn-Phong needs much higher exponent and drastically lower weight than Phong
      // because halfway vector activates over a much broader surface area
      let blinn_exponent = lighting.specular_power * 6.0;
      let spec_intensity = pow(max(0.0, dot(hit_normal, halfway)), blinn_exponent);
      // Very low weight (0.03 vs 0.2 for Phong) to match overall brightness contribution
      specular_light = lighting.light_color * spec_intensity * 0.03;
    }
    
    // Properly combine: albedo affects diffuse/ambient, specular is separate (direct reflection)
    final_color = hit_albedo * (diffuse_light + indirect_light) + specular_light;
  }

  // Apply tonemapping (skip for background to preserve colors)
  if (is_hit) {
    final_color = reinhard_tonemap(final_color, uniforms.cam_exposure);
  }

  // Write AOVs if enabled
  if (aov_enabled(AOV_ALBEDO_BIT)) {
    let albedo_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(hit_albedo, 1.0), is_hit);
    textureStore(aov_albedo, pixel_coord, albedo_val);
  }

  if (aov_enabled(AOV_NORMAL_BIT)) {
    let normal_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(hit_normal, 1.0), is_hit);
    textureStore(aov_normal, pixel_coord, normal_val);
  }

  if (aov_enabled(AOV_DEPTH_BIT)) {
    let depth_val: f32 = select(bitcast<f32>(0x7fc00000u), t_best, is_hit); // qNaN for miss
    textureStore(aov_depth, pixel_coord, vec4<f32>(depth_val, 0.0, 0.0, 0.0));
  }

  if (aov_enabled(AOV_DIRECT_BIT)) {
    // Direct lighting is diffuse + specular combined
    let total_direct = diffuse_light + specular_light;
    let direct_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(total_direct, 1.0), is_hit);
    textureStore(aov_direct, pixel_coord, direct_val);
  }

  if (aov_enabled(AOV_INDIRECT_BIT)) {
    let indirect_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(indirect_light, 1.0), is_hit);
    textureStore(aov_indirect, pixel_coord, indirect_val);
  }

  if (aov_enabled(AOV_EMISSION_BIT)) {
    // No emission in this simple implementation
    textureStore(aov_emission, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
  }

  if (aov_enabled(AOV_VISIBILITY_BIT)) {
    let visibility_val: f32 = select(0.0, 1.0, is_hit);
    textureStore(aov_visibility, pixel_coord, vec4<f32>(visibility_val, 0.0, 0.0, 1.0));
  }

  // Write final output
  textureStore(out_tex, pixel_coord, vec4<f32>(final_color, 1.0));
}
