use crate::viewer::terrain;
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

fn requested_manual_sun(
    current_direction: [f32; 3],
    current_intensity: f32,
    azimuth: Option<f32>,
    elevation: Option<f32>,
    intensity: Option<f32>,
) -> Option<(f32, f32, f32)> {
    if azimuth.is_none() && elevation.is_none() && intensity.is_none() {
        return None;
    }
    let direction = glam::Vec3::from_array(current_direction).normalize_or_zero();
    let current_azimuth = direction
        .x
        .atan2(direction.z)
        .to_degrees()
        .rem_euclid(360.0);
    let current_elevation = direction.y.clamp(-1.0, 1.0).asin().to_degrees();
    Some((
        azimuth.unwrap_or(current_azimuth),
        elevation.unwrap_or(current_elevation).clamp(-90.0, 90.0),
        intensity.unwrap_or(current_intensity).max(0.0),
    ))
}

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::LoadTerrain(path) => {
            println!("[XYZZY_TERRAIN] LoadTerrain handler entry, path={}", path);
            if let Err(err) = terrain::ViewerTerrainScene::preflight_terrain_path(path) {
                eprintln!("[terrain] rejected before scene/resource mutation: {err}");
                viewer.reject_command(format!(
                    "terrain_metadata_preflight_rejected: path={path} error={err} unchanged_state=true"
                ));
                return true;
            }
            if viewer.terrain_viewer.is_none() {
                eprintln!("[DEBUG LoadTerrain] Staging new terrain_viewer");
                let mut candidate = match terrain::ViewerTerrainScene::new(
                    std::sync::Arc::clone(&viewer.device),
                    std::sync::Arc::clone(&viewer.queue),
                    viewer.config.format,
                ) {
                    Ok(scene) => scene,
                    Err(e) => {
                        eprintln!("[terrain] Failed to create viewer: {}", e);
                        viewer.reject_command(format!("terrain_scene_construction_failed: {e}"));
                        return true;
                    }
                };
                if let Err(error) = candidate.load_terrain(path) {
                    viewer.reject_command(format!(
                        "terrain_execution_failed: path={path} error={error} unchanged_state=true"
                    ));
                    return true;
                }
                viewer.terrain_viewer = Some(candidate);
                println!("[terrain] Loaded: {}", path);
                eprintln!("[DEBUG LoadTerrain] staged terrain_viewer published");
            } else {
                eprintln!("[DEBUG LoadTerrain] terrain_viewer already exists");
                if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                    if let Err(error) = terrain_viewer.load_terrain(path) {
                        viewer.reject_command(format!(
                            "terrain_execution_failed: path={path} error={error} unchanged_state=true"
                        ));
                        return true;
                    }
                    println!("[terrain] Loaded: {}", path);
                }
            }
            // A loaded scene owns fresh terrain lighting state; restore the live sun.
            viewer.sync_terrain_sun_to_lit();
            viewer.astro_terrain_revision = viewer.astro_observation_revision;
            true
        }
        ViewerCmd::SetTerrainCamera {
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
            target,
        } => {
            let anchor = viewer.camera_anchor;
            let outcome = if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                match terrain_viewer
                    .set_camera(*phi_deg, *theta_deg, *radius, *fov_deg, *target, &anchor)
                {
                    Ok(()) => {
                        println!(
                            "[terrain] Camera: phi={:.1}° theta={:.1}° r={:.1} fov={:.1}° target={:?}",
                            phi_deg, theta_deg, radius, fov_deg, target
                        );
                        Ok(())
                    }
                    Err(err) => {
                        eprintln!("[terrain] camera rejected transactionally: {err}");
                        Err(err.to_string())
                    }
                }
            } else {
                Err("terrain camera requires a loaded terrain".to_string())
            };
            if let Err(error) = outcome {
                viewer.reject_command(error);
            }
            true
        }
        ViewerCmd::SetTerrainSun {
            azimuth_deg,
            elevation_deg,
            intensity,
            source,
        } => {
            viewer.apply_manual_sun(*azimuth_deg, *elevation_deg, Some(*intensity));
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_sun(
                    *azimuth_deg,
                    *elevation_deg,
                    *intensity,
                    source.as_deref().unwrap_or("manual_angles"),
                );
            }
            println!(
                "[terrain] Sun: az={:.1}° el={:.1}° int={:.2}",
                azimuth_deg, elevation_deg, intensity
            );
            true
        }
        ViewerCmd::SetTerrain {
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
        } => {
            let anchor = viewer.camera_anchor;
            let mut manual_sun = requested_manual_sun(
                viewer.lit_sun_direction_ws,
                viewer.lit_sun_intensity,
                *sun_azimuth,
                *sun_elevation,
                *sun_intensity,
            );
            if manual_sun.is_some_and(|sun| ![sun.0, sun.1, sun.2].into_iter().all(f32::is_finite))
            {
                viewer.reject_command(
                    "terrain_sun_rejected: non-finite manual sun unchanged_state=true",
                );
                return true;
            }
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                if let Some(terrain) = terrain_viewer.terrain.as_mut() {
                    let old_default_target = terrain.default_camera_target();
                    let target_was_default =
                        terrain.cam_target.abs_diff_eq(old_default_target, 0.01);

                    let candidate_phi = phi.as_ref().copied().unwrap_or(terrain.cam_phi_deg);
                    let candidate_theta = theta
                        .as_ref()
                        .map(|value| value.clamp(0.0, 85.0))
                        .unwrap_or(terrain.cam_theta_deg);
                    let candidate_radius = radius
                        .as_ref()
                        .map(|value| value.clamp(100.0, 50000.0))
                        .unwrap_or(terrain.cam_radius);
                    let candidate_fov = fov
                        .as_ref()
                        .map(|value| value.clamp(10.0, 120.0))
                        .unwrap_or(terrain.cam_fov_deg);
                    let candidate_zscale = zscale
                        .as_ref()
                        .map(|value| value.max(0.01))
                        .unwrap_or(terrain.z_scale);
                    let mut candidate_target = target
                        .as_ref()
                        .map(|value| glam::DVec3::from(*value))
                        .unwrap_or(terrain.cam_target);
                    if target.is_none() && zscale.is_some() && target_was_default {
                        candidate_target.y =
                            f64::from(terrain.height_range() * candidate_zscale * 0.5);
                    }
                    let candidate_sun_azimuth = sun_azimuth
                        .as_ref()
                        .copied()
                        .unwrap_or(terrain.sun_azimuth_deg);
                    let candidate_sun_elevation = sun_elevation
                        .as_ref()
                        .map(|value| value.clamp(-90.0, 90.0))
                        .unwrap_or(terrain.sun_elevation_deg);
                    let candidate_sun_intensity = sun_intensity
                        .as_ref()
                        .map(|value| value.max(0.0))
                        .unwrap_or(terrain.sun_intensity);
                    let candidate_ambient = ambient
                        .as_ref()
                        .map(|value| value.clamp(0.0, 1.0))
                        .unwrap_or(terrain.ambient);
                    let candidate_shadow = shadow
                        .as_ref()
                        .map(|value| value.clamp(0.0, 1.0))
                        .unwrap_or(terrain.shadow_intensity);
                    let candidate_water_level =
                        water_level.as_ref().copied().unwrap_or(terrain.water_level);
                    let finite = [
                        candidate_sun_azimuth,
                        candidate_sun_elevation,
                        candidate_sun_intensity,
                        candidate_ambient,
                        candidate_zscale,
                        candidate_shadow,
                        candidate_water_level,
                    ]
                    .into_iter()
                    .all(f32::is_finite)
                        && background
                            .as_ref()
                            .map_or(true, |value| value.iter().copied().all(f32::is_finite))
                        && water_color
                            .as_ref()
                            .map_or(true, |value| value.iter().copied().all(f32::is_finite));
                    if !finite
                        || terrain
                            .validate_camera_state(
                                &anchor,
                                candidate_phi,
                                candidate_theta,
                                candidate_radius,
                                candidate_fov,
                                candidate_target,
                            )
                            .is_err()
                    {
                        eprintln!("[terrain] SetTerrain rejected transactionally");
                        viewer.command_error = Some(
                            "terrain_state_rejected: invalid finite/residual contract unchanged_state=true"
                                .to_string(),
                        );
                        return true;
                    }

                    terrain.cam_phi_deg = candidate_phi;
                    terrain.cam_theta_deg = candidate_theta;
                    terrain.cam_radius = candidate_radius;
                    terrain.cam_fov_deg = candidate_fov;
                    terrain.cam_target = candidate_target;
                    terrain.sun_azimuth_deg = candidate_sun_azimuth;
                    terrain.sun_elevation_deg = candidate_sun_elevation;
                    if sun_azimuth.is_some() || sun_elevation.is_some() {
                        terrain.sun_source = "manual_angles".to_string();
                    }
                    terrain.sun_intensity = candidate_sun_intensity;
                    if sun_azimuth.is_some() || sun_elevation.is_some() || sun_intensity.is_some() {
                        manual_sun = Some((
                            candidate_sun_azimuth,
                            candidate_sun_elevation,
                            candidate_sun_intensity,
                        ));
                    }
                    terrain.ambient = candidate_ambient;
                    terrain.z_scale = candidate_zscale;
                    terrain.shadow_intensity = candidate_shadow;
                    terrain.water_level = candidate_water_level;
                    if let Some(value) = background {
                        terrain.background_color = *value;
                    }
                    if let Some(value) = water_color {
                        terrain.water_color = *value;
                    }
                }
                if let Some(params) = terrain_viewer.get_params() {
                    println!("[terrain] {}", params);
                }
            }
            if let Some((azimuth, elevation, intensity)) = manual_sun {
                viewer.apply_manual_sun(azimuth, elevation, Some(intensity));
            }
            true
        }
        ViewerCmd::GetTerrainParams => {
            if let Some(ref terrain_viewer) = viewer.terrain_viewer {
                if let Some(params) = terrain_viewer.get_params() {
                    println!("[terrain] {}", params);
                }
            }
            true
        }
        ViewerCmd::SetTerrainScatter { batches } => {
            #[cfg(feature = "enable-gpu-instancing")]
            {
                if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                    match terrain_viewer.set_scatter_batches_from_configs(batches) {
                        Ok(()) => {
                            println!("[terrain] scatter batches set: {}", batches.len());
                        }
                        Err(e) => eprintln!("[terrain] Failed to set scatter batches: {e:#}"),
                    }
                } else {
                    eprintln!("[terrain] Load terrain before setting scatter batches");
                }
            }
            #[cfg(not(feature = "enable-gpu-instancing"))]
            {
                let _ = batches;
                eprintln!(
                    "[terrain] Scatter batches require Cargo feature 'enable-gpu-instancing'"
                );
            }
            true
        }
        ViewerCmd::ClearTerrainScatter => {
            #[cfg(feature = "enable-gpu-instancing")]
            {
                if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                    terrain_viewer.clear_scatter_batches();
                    println!("[terrain] scatter batches cleared");
                }
            }
            #[cfg(not(feature = "enable-gpu-instancing"))]
            {
                eprintln!(
                    "[terrain] Scatter batches require Cargo feature 'enable-gpu-instancing'"
                );
            }
            true
        }
        ViewerCmd::SetTerrainPbr {
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
        } => {
            let update_result = if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_terrain_pbr(
                    *enabled,
                    hdr_path.clone(),
                    *ibl_intensity,
                    *hdr_rotate_deg,
                    shadow_technique.clone(),
                    *shadow_map_res,
                    *exposure,
                    *msaa,
                    *normal_strength,
                    height_ao.as_ref().clone(),
                    sun_visibility.as_ref().clone(),
                    materials.as_ref().clone(),
                    vector_overlay.as_ref().clone(),
                    tonemap.clone(),
                    lens_effects.clone(),
                    dof.as_ref().clone(),
                    motion_blur.clone(),
                    volumetrics.as_ref().clone(),
                    denoise.clone(),
                    *debug_mode,
                )
            } else {
                Err(anyhow::anyhow!(
                    "terrain PBR configuration requires a loaded terrain"
                ))
            };
            if let Err(error) = update_result {
                viewer.reject_command(format!(
                    "terrain_pbr_configuration_rejected: error={error} unchanged_state=true"
                ));
                return true;
            }

            if let Some(ref cfg) = sky {
                viewer.sky_enabled = cfg.enabled;
                if cfg.enabled {
                    viewer.sky_turbidity = cfg.turbidity;
                    viewer.sky_ground_albedo = cfg.ground_albedo;
                    viewer.sky_exposure = cfg.sky_exposure;
                    viewer.sky_sun_intensity = cfg.sun_intensity;
                    println!(
                        "[terrain] Sky enabled: turbidity={:.1} ground_albedo={:.2} exposure={:.2}",
                        cfg.turbidity, cfg.ground_albedo, cfg.sky_exposure
                    );
                }
            }
            true
        }
        ViewerCmd::LoadOverlay {
            name,
            path,
            extent,
            opacity,
            z_order,
        } => {
            println!(
                "[overlay] LoadOverlay command received: name='{}' path='{}'",
                name, path
            );
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                let opacity = opacity.unwrap_or(1.0);
                let z_order = z_order.unwrap_or(0);
                println!("[overlay] terrain_viewer exists, calling add_overlay_image...");
                match terrain_viewer.add_overlay_image(
                    name,
                    std::path::Path::new(path),
                    extent.clone(),
                    opacity,
                    crate::viewer::terrain::BlendMode::Normal,
                    z_order,
                ) {
                    Ok(id) => println!("[overlay] Loaded '{}' from {} (id={})", name, path, id),
                    Err(e) => eprintln!("[overlay] Failed to load '{}': {}", name, e),
                }
            } else {
                eprintln!(
                    "[overlay] No terrain loaded - load terrain first (terrain_viewer is None)"
                );
            }
            true
        }
        ViewerCmd::RemoveOverlay { id } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                if terrain_viewer.remove_overlay(*id) {
                    println!("[overlay] Removed overlay id={}", id);
                } else {
                    eprintln!("[overlay] Overlay id={} not found", id);
                }
            }
            true
        }
        ViewerCmd::SetOverlayVisible { id, visible } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_visible(*id, *visible);
                println!("[overlay] id={} visible={}", id, visible);
            }
            true
        }
        ViewerCmd::SetOverlayOpacity { id, opacity } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_opacity(*id, *opacity);
                println!("[overlay] id={} opacity={:.2}", id, opacity);
            }
            true
        }
        ViewerCmd::SetGlobalOverlayOpacity { opacity } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_global_overlay_opacity(*opacity);
                println!("[overlay] global opacity={:.2}", opacity);
            }
            true
        }
        ViewerCmd::SetOverlaysEnabled { enabled } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlays_enabled(*enabled);
                println!("[overlay] enabled={}", enabled);
            }
            true
        }
        ViewerCmd::SetOverlaySolid { solid } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_solid(*solid);
                println!("[overlay] solid={}", solid);
            }
            true
        }
        ViewerCmd::SetOverlayPreserveColors { preserve_colors } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_preserve_colors(*preserve_colors);
                println!("[overlay] preserve_colors={}", preserve_colors);
            }
            true
        }
        ViewerCmd::ListOverlays => {
            if let Some(ref terrain_viewer) = viewer.terrain_viewer {
                let ids = terrain_viewer.list_overlays();
                if ids.is_empty() {
                    println!("[overlay] No overlays loaded");
                } else {
                    println!("[overlay] Loaded overlays: {:?}", ids);
                }
            } else {
                println!("[overlay] No terrain loaded");
            }
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::requested_manual_sun;

    #[test]
    fn sun_bearing_set_terrain_has_a_manual_transition_without_loaded_terrain() {
        let transition = requested_manual_sun([1.0, 0.0, 0.0], 2.0, None, Some(30.0), None)
            .expect("sun-bearing command must transition out of observation replay");
        assert!((transition.0 - 90.0).abs() < 1.0e-5);
        assert_eq!(transition.1, 30.0);
        assert_eq!(transition.2, 2.0);
        assert!(requested_manual_sun([0.0, 1.0, 0.0], 1.0, None, None, None).is_none());
    }
}
