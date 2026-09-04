use crate::viewer::viewer_enums::{
    ViewerCmd, ViewerDenoiseConfig, ViewerDensityVolumeConfig, ViewerDofConfig,
    ViewerHeightAoConfig, ViewerLensEffectsConfig, ViewerMaterialLayerConfig,
    ViewerMotionBlurConfig, ViewerSkyConfig, ViewerSunVisConfig, ViewerTerrainScatterBatchConfig,
    ViewerTerrainScatterBlendConfig, ViewerTerrainScatterContactConfig,
    ViewerTerrainScatterLevelConfig, ViewerTonemapConfig, ViewerVectorOverlayConfig,
    ViewerVolumetricsConfig,
};

#[cfg(feature = "enable-gpu-instancing")]
use super::super::payloads::IpcScatterWind;
use super::super::payloads::{
    IpcDenoiseConfig, IpcDensityVolumeConfig, IpcDofConfig, IpcHeightAoConfig,
    IpcLensEffectsConfig, IpcMaterialLayerConfig, IpcMotionBlurConfig, IpcSkyConfig,
    IpcSunVisConfig, IpcTerrainScatterBatch, IpcTerrainScatterBlend, IpcTerrainScatterContact,
    IpcTerrainScatterLevel, IpcTonemapConfig, IpcVectorOverlayConfig, IpcVolumetricsConfig,
};
use super::super::request::IpcRequest;

pub(super) fn to_viewer_cmd(req: &IpcRequest) -> Result<Option<ViewerCmd>, String> {
    match req {
        IpcRequest::LoadTerrain { path } => {
            crate::viewer::terrain::ViewerTerrainScene::preflight_terrain_path(path)
                .map_err(|err| err.to_string())?;
            Ok(Some(ViewerCmd::LoadTerrain(path.clone())))
        }
        IpcRequest::SetTerrainCamera {
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
            target,
        } => Ok(Some(ViewerCmd::SetTerrainCamera {
            phi_deg: *phi_deg,
            theta_deg: *theta_deg,
            radius: *radius,
            fov_deg: *fov_deg,
            target: *target,
        })),
        IpcRequest::SetTerrainSun {
            azimuth_deg,
            elevation_deg,
            intensity,
            source,
        } => {
            if !matches!(
                source.as_deref(),
                None | Some("manual_angles" | "solar_time")
            ) {
                return Err("terrain sun source must be 'manual_angles' or 'solar_time'".into());
            }
            Ok(Some(ViewerCmd::SetTerrainSun {
                azimuth_deg: *azimuth_deg,
                elevation_deg: *elevation_deg,
                intensity: *intensity,
                source: source.clone(),
            }))
        }
        IpcRequest::SetTerrain {
            phi,
            theta,
            radius,
            fov,
            sun_azimuth,
            sun_elevation,
            sun_intensity,
            ambient,
            zscale,
            shadow,
            background,
            water_level,
            water_color,
            target,
        } => Ok(Some(ViewerCmd::SetTerrain {
            phi: *phi,
            theta: *theta,
            radius: *radius,
            fov: *fov,
            sun_azimuth: *sun_azimuth,
            sun_elevation: *sun_elevation,
            sun_intensity: *sun_intensity,
            ambient: *ambient,
            zscale: *zscale,
            shadow: *shadow,
            background: *background,
            water_level: *water_level,
            water_color: *water_color,
            target: *target,
        })),
        IpcRequest::GetTerrainParams => Ok(Some(ViewerCmd::GetTerrainParams)),
        IpcRequest::SetTerrainScatter { batches } => Ok(Some(ViewerCmd::SetTerrainScatter {
            batches: batches
                .iter()
                .enumerate()
                .map(|(batch_index, batch)| map_terrain_scatter_batch(batch, batch_index))
                .collect::<Result<Vec<_>, _>>()?,
        })),
        IpcRequest::ClearTerrainScatter => Ok(Some(ViewerCmd::ClearTerrainScatter)),
        IpcRequest::SetTerrainPbr {
            enabled,
            hdr_path,
            ibl_intensity,
            hdr_rotate_deg,
            shadow_technique,
            shadow_map_res,
            exposure,
            msaa,
            normal_strength,
            height_ao,
            sun_visibility,
            materials,
            vector_overlay,
            tonemap,
            dof,
            motion_blur,
            lens_effects,
            denoise,
            volumetrics,
            sky,
            debug_mode,
        } => Ok(Some(ViewerCmd::SetTerrainPbr {
            enabled: *enabled,
            hdr_path: hdr_path.clone(),
            ibl_intensity: *ibl_intensity,
            hdr_rotate_deg: *hdr_rotate_deg,
            shadow_technique: shadow_technique.clone(),
            shadow_map_res: *shadow_map_res,
            exposure: *exposure,
            msaa: *msaa,
            normal_strength: *normal_strength,
            height_ao: Box::new(height_ao.as_ref().map(map_height_ao)),
            sun_visibility: Box::new(sun_visibility.as_ref().map(map_sun_vis)),
            materials: Box::new(materials.as_ref().map(map_materials)),
            vector_overlay: Box::new(vector_overlay.as_ref().map(map_vector_overlay)),
            tonemap: tonemap.as_ref().map(map_tonemap),
            dof: Box::new(dof.as_ref().map(map_dof)),
            motion_blur: motion_blur.as_ref().map(map_motion_blur),
            lens_effects: lens_effects.as_ref().map(map_lens_effects),
            denoise: denoise.as_ref().map(map_denoise),
            volumetrics: Box::new(volumetrics.as_ref().map(map_volumetrics)),
            sky: sky.as_ref().map(map_sky),
            debug_mode: *debug_mode,
        })),
        _ => Ok(None),
    }
}

fn map_height_ao(config: &IpcHeightAoConfig) -> ViewerHeightAoConfig {
    ViewerHeightAoConfig {
        enabled: config.enabled.unwrap_or(false),
        directions: config.directions.unwrap_or(6),
        steps: config.steps.unwrap_or(16),
        max_distance: config.max_distance.unwrap_or(200.0),
        strength: config.strength.unwrap_or(1.0),
        resolution_scale: config.resolution_scale.unwrap_or(0.5),
    }
}

fn map_sun_vis(config: &IpcSunVisConfig) -> ViewerSunVisConfig {
    ViewerSunVisConfig {
        enabled: config.enabled.unwrap_or(false),
        mode: config.mode.clone().unwrap_or_else(|| "soft".to_string()),
        samples: config.samples.unwrap_or(4),
        steps: config.steps.unwrap_or(24),
        max_distance: config.max_distance.unwrap_or(400.0),
        softness: config.softness.unwrap_or(1.0),
        bias: config.bias.unwrap_or(0.01),
        resolution_scale: config.resolution_scale.unwrap_or(0.5),
    }
}

fn map_materials(config: &IpcMaterialLayerConfig) -> ViewerMaterialLayerConfig {
    ViewerMaterialLayerConfig {
        snow_enabled: config.snow_enabled.unwrap_or(false),
        snow_altitude_min: config.snow_altitude_min.unwrap_or(2500.0),
        snow_altitude_blend: config.snow_altitude_blend.unwrap_or(200.0),
        snow_slope_max: config.snow_slope_max.unwrap_or(45.0),
        rock_enabled: config.rock_enabled.unwrap_or(false),
        rock_slope_min: config.rock_slope_min.unwrap_or(45.0),
        wetness_enabled: config.wetness_enabled.unwrap_or(false),
        wetness_strength: config.wetness_strength.unwrap_or(0.3),
    }
}

fn map_vector_overlay(config: &IpcVectorOverlayConfig) -> ViewerVectorOverlayConfig {
    ViewerVectorOverlayConfig {
        depth_test: config.depth_test.unwrap_or(false),
        depth_bias: config.depth_bias.unwrap_or(0.001),
        halo_enabled: config.halo_enabled.unwrap_or(false),
        halo_width: config.halo_width.unwrap_or(2.0),
        halo_color: config.halo_color.unwrap_or([0.0, 0.0, 0.0, 0.5]),
    }
}

fn map_tonemap(config: &IpcTonemapConfig) -> ViewerTonemapConfig {
    ViewerTonemapConfig {
        operator: config
            .operator
            .clone()
            .unwrap_or_else(|| "aces".to_string()),
        white_point: config.white_point.unwrap_or(4.0),
        white_balance_enabled: config.white_balance_enabled.unwrap_or(false),
        temperature: config.temperature.unwrap_or(6500.0),
        tint: config.tint.unwrap_or(0.0),
    }
}

fn map_dof(config: &IpcDofConfig) -> ViewerDofConfig {
    ViewerDofConfig {
        enabled: config.enabled.unwrap_or(false),
        f_stop: config.f_stop.unwrap_or(5.6),
        focus_distance: config.focus_distance.unwrap_or(100.0),
        focal_length: config.focal_length.unwrap_or(50.0),
        tilt_pitch: config.tilt_pitch.unwrap_or(0.0),
        tilt_yaw: config.tilt_yaw.unwrap_or(0.0),
        quality: config
            .quality
            .clone()
            .unwrap_or_else(|| "medium".to_string()),
    }
}

fn map_motion_blur(config: &IpcMotionBlurConfig) -> ViewerMotionBlurConfig {
    ViewerMotionBlurConfig {
        enabled: config.enabled.unwrap_or(false),
        samples: config.samples.unwrap_or(8),
        shutter_open: config.shutter_open.unwrap_or(0.0),
        shutter_close: config.shutter_close.unwrap_or(0.5),
        cam_phi_delta: config.cam_phi_delta.unwrap_or(0.0),
        cam_theta_delta: config.cam_theta_delta.unwrap_or(0.0),
        cam_radius_delta: config.cam_radius_delta.unwrap_or(0.0),
    }
}

fn map_lens_effects(config: &IpcLensEffectsConfig) -> ViewerLensEffectsConfig {
    ViewerLensEffectsConfig {
        enabled: config.enabled.unwrap_or(false),
        distortion: config.distortion.unwrap_or(0.0),
        chromatic_aberration: config.chromatic_aberration.unwrap_or(0.0),
        vignette_strength: config.vignette_strength.unwrap_or(0.0),
        vignette_radius: config.vignette_radius.unwrap_or(0.7),
        vignette_softness: config.vignette_softness.unwrap_or(0.3),
    }
}

fn map_denoise(config: &IpcDenoiseConfig) -> ViewerDenoiseConfig {
    ViewerDenoiseConfig {
        enabled: config.enabled.unwrap_or(false),
        method: config
            .method
            .clone()
            .unwrap_or_else(|| "atrous".to_string()),
        iterations: config.iterations.unwrap_or(3),
        sigma_color: config.sigma_color.unwrap_or(0.1),
    }
}

fn map_volumetrics(config: &IpcVolumetricsConfig) -> ViewerVolumetricsConfig {
    ViewerVolumetricsConfig {
        enabled: config.enabled.unwrap_or(false),
        mode: config.mode.clone().unwrap_or_else(|| "uniform".to_string()),
        density: config.density.unwrap_or(0.01),
        height_falloff: config.height_falloff.unwrap_or(0.1),
        scattering: config.scattering.unwrap_or(0.5),
        absorption: config.absorption.unwrap_or(0.1),
        light_shafts: config.light_shafts.unwrap_or(false),
        shaft_intensity: config.shaft_intensity.unwrap_or(1.0),
        steps: config.steps.unwrap_or(32),
        half_res: config.half_res.unwrap_or(false),
        density_volumes: config
            .density_volumes
            .iter()
            .map(map_density_volume)
            .collect(),
    }
}

fn map_density_volume(config: &IpcDensityVolumeConfig) -> ViewerDensityVolumeConfig {
    ViewerDensityVolumeConfig {
        preset: config
            .preset
            .clone()
            .unwrap_or_else(|| "valley_fog".to_string()),
        center: config.center.unwrap_or([0.0, 0.0, 0.0]),
        size: config.size.unwrap_or([128.0, 64.0, 128.0]),
        resolution: config.resolution.unwrap_or([64, 32, 64]),
        density_scale: config.density_scale.unwrap_or(1.0),
        edge_softness: config.edge_softness.unwrap_or(0.25),
        noise_strength: config.noise_strength.unwrap_or(0.35),
        floor_offset: config.floor_offset.unwrap_or(0.0),
        ceiling: config.ceiling.unwrap_or(0.4),
        plume_spread: config.plume_spread.unwrap_or(0.35),
        wind: config.wind.unwrap_or([0.25, 1.0, 0.0]),
        seed: config.seed.unwrap_or(0),
    }
}

fn map_sky(config: &IpcSkyConfig) -> ViewerSkyConfig {
    ViewerSkyConfig {
        enabled: config.enabled.unwrap_or(false),
        turbidity: config.turbidity.unwrap_or(2.0),
        ground_albedo: config.ground_albedo.unwrap_or(0.3),
        sun_intensity: config.sun_intensity.unwrap_or(1.0),
        aerial_perspective: config.aerial_perspective.unwrap_or(true),
        sky_exposure: config.sky_exposure.unwrap_or(1.0),
    }
}

pub(super) fn map_terrain_scatter_batch(
    config: &IpcTerrainScatterBatch,
    batch_index: usize,
) -> Result<ViewerTerrainScatterBatchConfig, String> {
    Ok(ViewerTerrainScatterBatchConfig {
        name: config.name.clone(),
        color: config.color.unwrap_or([0.85, 0.85, 0.85, 1.0]),
        max_draw_distance: config.max_draw_distance,
        terrain_blend: config
            .terrain_blend
            .as_ref()
            .map(map_terrain_scatter_blend)
            .unwrap_or_default(),
        terrain_contact: config
            .terrain_contact
            .as_ref()
            .map(map_terrain_scatter_contact)
            .unwrap_or_default(),
        transforms: config.transforms.clone(),
        levels: config
            .levels
            .iter()
            .map(map_terrain_scatter_level)
            .collect(),
        #[cfg(feature = "enable-gpu-instancing")]
        wind: config
            .wind
            .as_ref()
            .map(|wind| {
                map_scatter_wind(wind).map_err(|e| format!("scatter batch {batch_index}: {e}"))
            })
            .transpose()?
            .unwrap_or_default(),
        #[cfg(feature = "enable-gpu-instancing")]
        hlod_config: config
            .hlod
            .as_ref()
            .map(|h| crate::terrain::scatter::HlodConfig {
                hlod_distance: h.hlod_distance,
                cluster_radius: h.cluster_radius,
                simplify_ratio: h.simplify_ratio,
            }),
    })
}

#[cfg(feature = "enable-gpu-instancing")]
fn map_scatter_wind(
    w: &IpcScatterWind,
) -> Result<crate::terrain::scatter::ScatterWindSettingsNative, String> {
    let settings = crate::terrain::scatter::ScatterWindSettingsNative {
        enabled: w.enabled,
        direction_deg: w.direction_deg,
        speed: w.speed,
        amplitude: w.amplitude,
        rigidity: w.rigidity,
        bend_start: w.bend_start,
        bend_extent: w.bend_extent,
        gust_strength: w.gust_strength,
        gust_frequency: w.gust_frequency,
        fade_start: w.fade_start,
        fade_end: w.fade_end,
    };
    settings.validate().map_err(|e| e.to_string())?;
    Ok(settings)
}

fn map_terrain_scatter_blend(config: &IpcTerrainScatterBlend) -> ViewerTerrainScatterBlendConfig {
    ViewerTerrainScatterBlendConfig {
        enabled: config.enabled.unwrap_or(false),
        bury_depth: config.bury_depth.unwrap_or(0.75),
        fade_distance: config.fade_distance.unwrap_or(2.5),
    }
}

fn map_terrain_scatter_contact(
    config: &IpcTerrainScatterContact,
) -> ViewerTerrainScatterContactConfig {
    ViewerTerrainScatterContactConfig {
        enabled: config.enabled.unwrap_or(false),
        distance: config.distance.unwrap_or(3.0),
        strength: config.strength.unwrap_or(0.35),
        vertical_weight: config.vertical_weight.unwrap_or(0.65),
    }
}

fn map_terrain_scatter_level(config: &IpcTerrainScatterLevel) -> ViewerTerrainScatterLevelConfig {
    ViewerTerrainScatterLevelConfig {
        mesh: crate::geometry::MeshBuffers {
            positions: config.positions.clone(),
            normals: config.normals.clone(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            indices: config.indices.clone(),
        },
        max_distance: config.max_distance,
    }
}
