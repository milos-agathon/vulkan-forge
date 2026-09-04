use super::types::ViewerTerrainPbrConfig;
use crate::viewer::viewer_enums::{
    ViewerDenoiseConfig, ViewerHeightAoConfig, ViewerMaterialLayerConfig, ViewerSunVisConfig,
    ViewerTonemapConfig, ViewerVectorOverlayConfig,
};
use std::path::PathBuf;

impl ViewerTerrainPbrConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn apply_updates(
        &mut self,
        enabled: Option<bool>,
        hdr_path: Option<String>,
        ibl_intensity: Option<f32>,
        hdr_rotate_deg: Option<f32>,
        shadow_technique: Option<String>,
        shadow_map_res: Option<u32>,
        exposure: Option<f32>,
        msaa: Option<u32>,
        normal_strength: Option<f32>,
        height_ao: Option<ViewerHeightAoConfig>,
        sun_visibility: Option<ViewerSunVisConfig>,
        materials: Option<ViewerMaterialLayerConfig>,
        vector_overlay: Option<ViewerVectorOverlayConfig>,
        tonemap: Option<ViewerTonemapConfig>,
        denoise: Option<ViewerDenoiseConfig>,
        debug_mode: Option<u32>,
    ) {
        if let Some(v) = enabled {
            self.enabled = v;
        }
        if let Some(p) = hdr_path {
            self.hdr_path = Some(PathBuf::from(p));
        }
        if let Some(v) = ibl_intensity {
            self.ibl_intensity = v.max(0.0);
        }
        if let Some(v) = hdr_rotate_deg {
            self.hdr_rotate_deg = v.rem_euclid(360.0);
        }
        if let Some(t) = shadow_technique {
            let t_lower = t.to_lowercase();
            let supported = t_lower == "none"
                || crate::lighting::types::ShadowTechnique::from_name(&t_lower).is_some_and(
                    |technique| technique != crate::lighting::types::ShadowTechnique::CSM,
                );
            if supported {
                self.shadow_technique = t_lower;
            }
        }
        if let Some(r) = shadow_map_res {
            self.shadow_map_res = r.clamp(512, 8192);
        }
        if let Some(e) = exposure {
            self.exposure = e.max(0.0);
        }
        if let Some(m) = msaa {
            self.msaa = match m {
                1 | 4 | 8 => m,
                _ => 1,
            };
        }
        if let Some(n) = normal_strength {
            self.normal_strength = n.clamp(0.0, 10.0);
        }
        if let Some(ao) = height_ao {
            self.height_ao.enabled = ao.enabled;
            self.height_ao.directions = ao.directions.clamp(4, 16);
            self.height_ao.steps = ao.steps.clamp(8, 64);
            self.height_ao.max_distance = ao.max_distance.max(0.0);
            self.height_ao.strength = ao.strength.clamp(0.0, 2.0);
            self.height_ao.resolution_scale = ao.resolution_scale.clamp(0.1, 1.0);
        }
        if let Some(sv) = sun_visibility {
            self.sun_visibility.enabled = sv.enabled;
            self.sun_visibility.mode = if sv.mode == "hard" {
                "hard".to_string()
            } else {
                "soft".to_string()
            };
            self.sun_visibility.samples = sv.samples.clamp(1, 16);
            self.sun_visibility.steps = sv.steps.clamp(8, 64);
            self.sun_visibility.max_distance = sv.max_distance.max(0.0);
            self.sun_visibility.softness = sv.softness.clamp(0.0, 4.0);
            self.sun_visibility.bias = sv.bias.clamp(0.0, 0.1);
            self.sun_visibility.resolution_scale = sv.resolution_scale.clamp(0.1, 1.0);
        }
        if let Some(mat) = materials {
            self.materials.snow_enabled = mat.snow_enabled;
            self.materials.snow_altitude_min = mat.snow_altitude_min.max(0.0);
            self.materials.snow_altitude_blend = mat.snow_altitude_blend.max(0.0);
            self.materials.snow_slope_max = mat.snow_slope_max.clamp(0.0, 90.0);
            self.materials.rock_enabled = mat.rock_enabled;
            self.materials.rock_slope_min = mat.rock_slope_min.clamp(0.0, 90.0);
            self.materials.wetness_enabled = mat.wetness_enabled;
            self.materials.wetness_strength = mat.wetness_strength.clamp(0.0, 1.0);
        }
        if let Some(vo) = vector_overlay {
            self.vector_overlay.depth_test = vo.depth_test;
            self.vector_overlay.depth_bias = vo.depth_bias.max(0.0);
            self.vector_overlay.halo_enabled = vo.halo_enabled;
            self.vector_overlay.halo_width = vo.halo_width.max(0.0);
            self.vector_overlay.halo_color = vo.halo_color;
        }
        if let Some(tm) = tonemap {
            let valid_ops = [
                "reinhard",
                "reinhard_extended",
                "aces",
                "uncharted2",
                "exposure",
            ];
            if valid_ops.contains(&tm.operator.to_lowercase().as_str()) {
                self.tonemap.operator = tm.operator.to_lowercase();
            }
            self.tonemap.white_point = tm.white_point.max(0.1);
            self.tonemap.white_balance_enabled = tm.white_balance_enabled;
            self.tonemap.temperature = tm.temperature.clamp(2000.0, 12000.0);
            self.tonemap.tint = tm.tint.clamp(-1.0, 1.0);
        }
        if let Some(dn) = denoise {
            self.denoise.enabled = dn.enabled;
            self.denoise.method = dn.method;
            self.denoise.iterations = dn.iterations.clamp(1, 8);
            self.denoise.sigma_color = dn.sigma_color.max(0.1);
        }
        if let Some(d) = debug_mode {
            self.debug_mode = d;
        }
    }

    pub fn apply_lens_effects(
        &mut self,
        enabled: bool,
        vignette: f32,
        radius: f32,
        softness: f32,
        distortion: f32,
        ca: f32,
    ) {
        self.lens_effects.enabled = enabled;
        self.lens_effects.vignette_strength = vignette.clamp(0.0, 1.0);
        self.lens_effects.vignette_radius = radius.clamp(0.1, 1.0);
        self.lens_effects.vignette_softness = softness.clamp(0.1, 1.0);
        self.lens_effects.distortion = distortion.clamp(-0.5, 0.5);
        self.lens_effects.chromatic_aberration = ca.clamp(0.0, 0.1);
    }

    pub fn apply_dof(
        &mut self,
        enabled: bool,
        f_stop: f32,
        focus_distance: f32,
        focal_length: f32,
        quality: &str,
        tilt_pitch_deg: f32,
        tilt_yaw_deg: f32,
    ) {
        self.dof.enabled = enabled;
        let clamped_f_stop = f_stop.clamp(1.4, 22.0);
        self.dof.f_stop = clamped_f_stop;
        self.dof.focus_distance = focus_distance.max(1.0);
        self.dof.focal_length = focal_length.clamp(10.0, 200.0);
        self.dof.quality = match quality.to_lowercase().as_str() {
            "low" => 4,
            "medium" => 8,
            "high" => 16,
            "ultra" => 32,
            _ => 8,
        };
        self.dof.tilt_pitch = tilt_pitch_deg.to_radians();
        self.dof.tilt_yaw = tilt_yaw_deg.to_radians();

        let base_strength = 50.0;
        let reference_f_stop = 1.4;
        self.dof.blur_strength = base_strength * (reference_f_stop / clamped_f_stop);
    }

    pub fn apply_motion_blur(
        &mut self,
        enabled: bool,
        samples: u32,
        shutter_open: f32,
        shutter_close: f32,
        cam_phi_delta: f32,
        cam_theta_delta: f32,
        cam_radius_delta: f32,
    ) {
        self.motion_blur.enabled = enabled;
        self.motion_blur.samples = samples.clamp(1, 64);
        self.motion_blur.shutter_open = shutter_open.clamp(0.0, 1.0);
        self.motion_blur.shutter_close = shutter_close.clamp(0.0, 1.0);
        self.motion_blur.cam_phi_delta = cam_phi_delta;
        self.motion_blur.cam_theta_delta = cam_theta_delta;
        self.motion_blur.cam_radius_delta = cam_radius_delta;
    }

    pub fn apply_volumetrics(
        &mut self,
        enabled: bool,
        mode: &str,
        density: f32,
        height_falloff: f32,
        scattering: f32,
        absorption: f32,
        light_shafts: bool,
        shaft_intensity: f32,
        steps: u32,
        half_res: bool,
        density_volumes: &[super::types::DensityVolumeConfig],
    ) {
        self.volumetrics.enabled = enabled;
        self.volumetrics.mode = mode.to_string();
        self.volumetrics.density = density.clamp(0.0, 0.5);
        self.volumetrics.height_falloff = height_falloff.clamp(0.0, 4.0);
        self.volumetrics.scattering = scattering.clamp(0.0, 1.0);
        self.volumetrics.absorption = absorption.clamp(0.0, 1.0);
        self.volumetrics.light_shafts = light_shafts;
        self.volumetrics.shaft_intensity = shaft_intensity.clamp(0.0, 10.0);
        self.volumetrics.steps = steps.clamp(8, 128);
        self.volumetrics.half_res = half_res;
        self.volumetrics.density_volumes = density_volumes.to_vec();
    }

    pub fn to_display_string(&self) -> String {
        let mut parts = vec![
            format!("PBR: {}", if self.enabled { "ON" } else { "OFF" }),
            format!(
                "shadow={} res={}",
                self.shadow_technique, self.shadow_map_res
            ),
            format!(
                "IBL={:.2} rot={:.1} exp={:.2}",
                self.ibl_intensity, self.hdr_rotate_deg, self.exposure
            ),
            format!("msaa={} normal={:.2}", self.msaa, self.normal_strength),
        ];
        if self.height_ao.enabled {
            parts.push(format!(
                "height_ao=ON dirs={} steps={}",
                self.height_ao.directions, self.height_ao.steps
            ));
        }
        if self.sun_visibility.enabled {
            parts.push(format!(
                "sun_vis={} samples={} steps={}",
                self.sun_visibility.mode, self.sun_visibility.samples, self.sun_visibility.steps
            ));
        }
        if self.materials.snow_enabled || self.materials.rock_enabled {
            parts.push(format!(
                "materials: snow={} rock={}",
                self.materials.snow_enabled, self.materials.rock_enabled
            ));
        }
        if self.vector_overlay.depth_test || self.vector_overlay.halo_enabled {
            parts.push(format!(
                "overlay: depth={} halo={}",
                self.vector_overlay.depth_test, self.vector_overlay.halo_enabled
            ));
        }
        if !self.volumetrics.density_volumes.is_empty() {
            parts.push(format!(
                "hetero_volumes={}",
                self.volumetrics.density_volumes.len()
            ));
        }
        parts.push(format!("tonemap={}", self.tonemap.operator));
        parts.join(" | ")
    }
}
