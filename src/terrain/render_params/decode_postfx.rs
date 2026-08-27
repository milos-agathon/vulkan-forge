use super::*;

pub(super) fn parse_tonemap_settings(params: &Bound<'_, PyAny>) -> TonemapSettingsNative {
    if let Ok(tm) = params.getattr("tonemap") {
        let operator_str: String = tm
            .getattr("operator")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| "aces".to_string());
        let operator_index = TonemapSettingsNative::operator_from_str(&operator_str);
        let white_point: f32 = tm
            .getattr("white_point")
            .and_then(|v| v.extract())
            .unwrap_or(4.0);
        let lut_enabled: bool = tm
            .getattr("lut_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let lut_path: Option<String> = tm
            .getattr("lut_path")
            .and_then(|v| v.extract())
            .ok()
            .flatten();
        let lut_strength: f32 = tm
            .getattr("lut_strength")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let white_balance_enabled: bool = tm
            .getattr("white_balance_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let temperature: f32 = tm
            .getattr("temperature")
            .and_then(|v| v.extract())
            .unwrap_or(6500.0);
        let tint: f32 = tm.getattr("tint").and_then(|v| v.extract()).unwrap_or(0.0);
        TonemapSettingsNative {
            operator_index,
            white_point,
            lut_enabled,
            lut_path,
            lut_strength,
            white_balance_enabled,
            temperature,
            tint,
        }
    } else {
        TonemapSettingsNative::default()
    }
}

pub(super) fn parse_aov_settings(params: &Bound<'_, PyAny>) -> AovSettingsNative {
    if let Ok(aov) = params.getattr("aov") {
        if !aov.is_none() {
            let enabled: bool = aov
                .getattr("enabled")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let albedo: bool = aov
                .getattr("albedo")
                .and_then(|v| v.extract())
                .unwrap_or(true);
            let normal: bool = aov
                .getattr("normal")
                .and_then(|v| v.extract())
                .unwrap_or(true);
            let depth: bool = aov
                .getattr("depth")
                .and_then(|v| v.extract())
                .unwrap_or(true);
            let transmittance: bool = aov
                .getattr("transmittance")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let in_scatter: bool = aov
                .getattr("in_scatter")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let cloud_shadow: bool = aov
                .getattr("cloud_shadow")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let optical_depth: bool = aov
                .getattr("optical_depth")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            // VERITAS: off unless the AovSettings dataclass carries the flag.
            let source_id: bool = aov
                .getattr("source_id")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let output_dir: Option<String> = aov
                .getattr("output_dir")
                .and_then(|v| {
                    if v.is_none() {
                        Ok(None)
                    } else {
                        v.extract().map(Some)
                    }
                })
                .unwrap_or(None);
            let format: String = aov
                .getattr("format")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "png".to_string());
            AovSettingsNative {
                enabled,
                albedo,
                normal,
                depth,
                transmittance,
                in_scatter,
                cloud_shadow,
                optical_depth,
                source_id,
                output_dir,
                format,
            }
        } else {
            AovSettingsNative::default()
        }
    } else {
        AovSettingsNative::default()
    }
}

pub(super) fn parse_screen_space_settings(params: &Bound<'_, PyAny>) -> ScreenSpaceSettingsNative {
    if let Ok(screen_space) = params.getattr("screen_space") {
        if !screen_space.is_none() {
            let ssao_enabled: bool = screen_space
                .getattr("ssao_enabled")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let ssgi_enabled: bool = screen_space
                .getattr("ssgi_enabled")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let ssr_enabled: bool = screen_space
                .getattr("ssr_enabled")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let taa_enabled: bool = screen_space
                .getattr("taa_enabled")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            let enabled: bool = screen_space
                .getattr("enabled")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            ScreenSpaceSettingsNative {
                enabled: enabled || ssao_enabled || ssgi_enabled || ssr_enabled || taa_enabled,
                ssao_enabled,
                ssao_radius: screen_space
                    .getattr("ssao_radius")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.5),
                ssao_intensity: screen_space
                    .getattr("ssao_intensity")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                ssgi_enabled,
                ssgi_intensity: screen_space
                    .getattr("ssgi_intensity")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                ssr_enabled,
                ssr_intensity: screen_space
                    .getattr("ssr_intensity")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                taa_enabled,
                temporal_alpha: screen_space
                    .getattr("temporal_alpha")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
            }
        } else {
            ScreenSpaceSettingsNative::default()
        }
    } else {
        ScreenSpaceSettingsNative::default()
    }
}

pub(super) fn parse_dof_settings(params: &Bound<'_, PyAny>) -> DofSettingsNative {
    if let Ok(dof) = params.getattr("dof") {
        if !dof.is_none() {
            let method_str: String = dof
                .getattr("method")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "gather".to_string());
            let quality_str: String = dof
                .getattr("quality")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "medium".to_string());
            let tilt_pitch_deg: f32 = dof
                .getattr("tilt_pitch")
                .and_then(|v| v.extract())
                .unwrap_or(0.0);
            let tilt_yaw_deg: f32 = dof
                .getattr("tilt_yaw")
                .and_then(|v| v.extract())
                .unwrap_or(0.0);
            DofSettingsNative {
                enabled: dof
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                f_stop: dof
                    .getattr("f_stop")
                    .and_then(|v| v.extract())
                    .unwrap_or(5.6),
                focus_distance: dof
                    .getattr("focus_distance")
                    .and_then(|v| v.extract())
                    .unwrap_or(100.0),
                focal_length: dof
                    .getattr("focal_length")
                    .and_then(|v| v.extract())
                    .unwrap_or(50.0),
                tilt_pitch: tilt_pitch_deg.to_radians(),
                tilt_yaw: tilt_yaw_deg.to_radians(),
                method: match method_str.as_str() {
                    "separable" => DofMethodNative::Separable,
                    _ => DofMethodNative::Gather,
                },
                quality: match quality_str.as_str() {
                    "low" => DofQualityNative::Low,
                    "high" => DofQualityNative::High,
                    "ultra" => DofQualityNative::Ultra,
                    _ => DofQualityNative::Medium,
                },
                show_coc: dof
                    .getattr("show_coc")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                debug_mode: dof
                    .getattr("debug_mode")
                    .and_then(|v| v.extract())
                    .unwrap_or(0),
            }
        } else {
            DofSettingsNative::default()
        }
    } else {
        DofSettingsNative::default()
    }
}

pub(super) fn parse_motion_blur_settings(params: &Bound<'_, PyAny>) -> MotionBlurSettingsNative {
    if let Ok(mb) = params.getattr("motion_blur") {
        if !mb.is_none() {
            let seed: Option<u64> = mb
                .getattr("seed")
                .and_then(|v| {
                    if v.is_none() {
                        Ok(None)
                    } else {
                        v.extract().map(Some)
                    }
                })
                .unwrap_or(None);
            MotionBlurSettingsNative {
                enabled: mb
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                samples: mb.getattr("samples").and_then(|v| v.extract()).unwrap_or(8),
                shutter_open: mb
                    .getattr("shutter_open")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                shutter_close: mb
                    .getattr("shutter_close")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.5),
                cam_phi_delta: mb
                    .getattr("cam_phi_delta")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                cam_theta_delta: mb
                    .getattr("cam_theta_delta")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                cam_radius_delta: mb
                    .getattr("cam_radius_delta")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                seed,
            }
        } else {
            MotionBlurSettingsNative::default()
        }
    } else {
        MotionBlurSettingsNative::default()
    }
}

pub(super) fn parse_lens_effects_settings(params: &Bound<'_, PyAny>) -> LensEffectsSettingsNative {
    if let Ok(le) = params.getattr("lens_effects") {
        if !le.is_none() {
            LensEffectsSettingsNative {
                enabled: le
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                distortion: le
                    .getattr("distortion")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                chromatic_aberration: le
                    .getattr("chromatic_aberration")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                vignette_strength: le
                    .getattr("vignette_strength")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                vignette_radius: le
                    .getattr("vignette_radius")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.7),
                vignette_softness: le
                    .getattr("vignette_softness")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.3),
            }
        } else {
            LensEffectsSettingsNative::default()
        }
    } else {
        LensEffectsSettingsNative::default()
    }
}

pub(super) fn parse_denoise_settings(params: &Bound<'_, PyAny>) -> DenoiseSettingsNative {
    if let Ok(dn) = params.getattr("denoise") {
        if !dn.is_none() {
            let method_str: String = dn
                .getattr("method")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "atrous".to_string());
            DenoiseSettingsNative {
                enabled: dn
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                method: match method_str.as_str() {
                    "oidn" => DenoiseMethodNative::Oidn,
                    "none" => DenoiseMethodNative::None,
                    _ => DenoiseMethodNative::Atrous,
                },
                iterations: dn
                    .getattr("iterations")
                    .and_then(|v| v.extract())
                    .unwrap_or(3),
                sigma_color: dn
                    .getattr("sigma_color")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                sigma_normal: dn
                    .getattr("sigma_normal")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                sigma_depth: dn
                    .getattr("sigma_depth")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                edge_stopping: dn
                    .getattr("edge_stopping")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
            }
        } else {
            DenoiseSettingsNative::default()
        }
    } else {
        DenoiseSettingsNative::default()
    }
}
