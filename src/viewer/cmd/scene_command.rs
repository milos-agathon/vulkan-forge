use std::path::{Path, PathBuf};

use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::Snapshot(path) => {
            let resolved = normalize_snapshot_path(path.clone());
            viewer.snapshot_request = Some(resolved);
            true
        }
        ViewerCmd::LoadObj(path) => {
            match crate::io::obj_read::import_obj(path) {
                Ok(obj) => {
                    if let Err(e) = viewer.upload_mesh(&obj.mesh) {
                        eprintln!("Failed to upload OBJ mesh: {}", e);
                    } else {
                        if let Some(mat) = obj.materials.first() {
                            if let Some(tex_rel) = &mat.diffuse_texture {
                                if let Some(base) = Path::new(path).parent() {
                                    let tex_path = base.join(tex_rel);
                                    let _ = viewer.load_albedo_texture(tex_path.as_path());
                                }
                            }
                        }
                        println!("Loaded OBJ geometry: {}", path);
                    }
                }
                Err(e) => eprintln!("OBJ import failed: {}", e),
            }
            true
        }
        ViewerCmd::LoadGltf(path) => {
            match crate::io::gltf_read::import_gltf_to_mesh(path) {
                Ok(mesh) => {
                    if let Err(e) = viewer.upload_mesh(&mesh) {
                        eprintln!("Failed to upload glTF mesh: {}", e);
                    }
                }
                Err(e) => eprintln!("glTF import failed: {}", e),
            }
            true
        }
        ViewerCmd::SetViz(mode) => {
            let next = match mode.as_str() {
                "material" | "mat" => crate::viewer::viewer_enums::VizMode::Material,
                "normal" | "normals" => crate::viewer::viewer_enums::VizMode::Normal,
                "depth" => crate::viewer::viewer_enums::VizMode::Depth,
                "gi" => crate::viewer::viewer_enums::VizMode::Gi,
                "lit" => crate::viewer::viewer_enums::VizMode::Lit,
                _ => {
                    eprintln!("Unknown viz mode: {}", mode);
                    viewer.viz_mode
                }
            };
            viewer.viz_mode = next;
            true
        }
        ViewerCmd::SetGiViz(mode) => {
            viewer.gi_viz_mode = *mode;
            viewer.viz_mode = match mode {
                crate::cli::gi_types::GiVizMode::None => crate::viewer::viewer_enums::VizMode::Lit,
                _ => crate::viewer::viewer_enums::VizMode::Gi,
            };
            true
        }
        ViewerCmd::QueryGiViz => {
            let name = match viewer.gi_viz_mode {
                crate::cli::gi_types::GiVizMode::None => "none",
                crate::cli::gi_types::GiVizMode::Composite => "composite",
                crate::cli::gi_types::GiVizMode::Ao => "ao",
                crate::cli::gi_types::GiVizMode::Ssgi => "ssgi",
                crate::cli::gi_types::GiVizMode::Ssr => "ssr",
            };
            println!("viz-gi = {}", name);
            true
        }
        ViewerCmd::LoadSsrPreset => {
            match viewer.apply_ssr_scene_preset() {
                Ok(_) => println!("[SSR] Loaded scene preset"),
                Err(e) => eprintln!("[SSR] Failed to load preset: {}", e),
            }
            true
        }
        ViewerCmd::SetLitSun(v) => {
            viewer.lit_sun_intensity = (*v).max(0.0);
            viewer.update_lit_uniform();
            viewer.sync_terrain_sun_to_lit();
            true
        }
        ViewerCmd::SetLitIbl(v) => {
            viewer.lit_ibl_intensity = (*v).max(0.0);
            viewer.lit_use_ibl = viewer.lit_ibl_intensity > 0.0;
            viewer.update_lit_uniform();
            true
        }
        ViewerCmd::SetLitBrdf(idx) => {
            viewer.lit_brdf = *idx;
            viewer.update_lit_uniform();
            true
        }
        ViewerCmd::SetLitRough(v) => {
            viewer.lit_roughness = v.clamp(0.0, 1.0);
            viewer.update_lit_uniform();
            true
        }
        ViewerCmd::SetLitDebug(mode) => {
            viewer.lit_debug_mode = match mode {
                1 | 2 => *mode,
                _ => 0,
            };
            viewer.update_lit_uniform();
            true
        }
        ViewerCmd::CaptureP51Cornell => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P51CornellSplit);
            println!("[P5.1] capture: Cornell OFF/ON split queued");
            true
        }
        ViewerCmd::CaptureP51Grid => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P51AoGrid);
            println!("[P5.1] capture: AO buffers grid queued");
            true
        }
        ViewerCmd::CaptureP51Sweep => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P51ParamSweep);
            println!("[P5.1] capture: AO parameter sweep queued");
            true
        }
        ViewerCmd::CaptureP52SsgiCornell => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P52SsgiCornell);
            println!("[P5.2] capture: SSGI Cornell split queued");
            true
        }
        ViewerCmd::CaptureP52SsgiTemporal => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P52SsgiTemporal);
            println!("[P5.2] capture: SSGI temporal compare queued");
            true
        }
        ViewerCmd::CaptureP53SsrGlossy => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P53SsrGlossy);
            println!("[P5.3] capture: SSR glossy spheres queued");
            true
        }
        ViewerCmd::CaptureP53SsrThickness => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P53SsrThickness);
            println!("[P5.3] capture: SSR thickness ablation queued");
            true
        }
        ViewerCmd::CaptureP54GiStack => {
            viewer
                .pending_captures
                .push_back(crate::viewer::viewer_enums::CaptureKind::P54GiStack);
            println!("[P5.4] capture: GI stack ablation queued");
            true
        }
        ViewerCmd::SkyToggle(on) => {
            viewer.sky_enabled = *on;
            true
        }
        ViewerCmd::SkySetModel(id) => {
            viewer.sky_model_id = *id;
            viewer.sky_enabled = true;
            true
        }
        ViewerCmd::SkySetTurbidity(t) => {
            viewer.sky_turbidity = t.clamp(1.0, 10.0);
            true
        }
        ViewerCmd::SkySetGround(a) => {
            viewer.sky_ground_albedo = a.clamp(0.0, 1.0);
            true
        }
        ViewerCmd::SkySetExposure(e) => {
            viewer.sky_exposure = (*e).max(0.0);
            true
        }
        ViewerCmd::SkySetSunIntensity(i) => {
            viewer.sky_sun_intensity = (*i).max(0.0);
            true
        }
        ViewerCmd::FogToggle(on) => {
            viewer.fog_enabled = *on;
            true
        }
        ViewerCmd::FogSetDensity(v) => {
            viewer.fog_density = (*v).max(0.0);
            true
        }
        ViewerCmd::FogSetG(v) => {
            viewer.fog_g = v.clamp(-0.999, 0.999);
            true
        }
        ViewerCmd::FogSetSteps(v) => {
            viewer.fog_steps = (*v).max(1);
            true
        }
        ViewerCmd::FogSetShadow(on) => {
            viewer.fog_use_shadows = *on;
            true
        }
        ViewerCmd::FogSetTemporal(v) => {
            viewer.fog_temporal_alpha = v.clamp(0.0, 0.9);
            true
        }
        ViewerCmd::SetFogMode(mode) => {
            viewer.fog_mode = if *mode != 0 {
                crate::viewer::viewer_enums::FogMode::Froxels
            } else {
                crate::viewer::viewer_enums::FogMode::Raymarch
            };
            true
        }
        ViewerCmd::FogHalf(on) => {
            viewer.fog_half_res_enabled = *on;
            true
        }
        ViewerCmd::FogEdges(on) => {
            viewer.fog_bilateral = *on;
            true
        }
        ViewerCmd::FogUpsigma(s) => {
            viewer.fog_upsigma = (*s).max(0.0);
            true
        }
        ViewerCmd::FogPreset(preset) => {
            match preset {
                0 => {
                    viewer.fog_steps = 32;
                    viewer.fog_temporal_alpha = 0.7;
                    viewer.fog_density = 0.02;
                }
                1 => {
                    viewer.fog_steps = 64;
                    viewer.fog_temporal_alpha = 0.6;
                    viewer.fog_density = 0.04;
                }
                _ => {
                    viewer.fog_steps = 96;
                    viewer.fog_temporal_alpha = 0.5;
                    viewer.fog_density = 0.06;
                }
            }
            true
        }
        ViewerCmd::HudToggle(on) => {
            viewer.hud_enabled = *on;
            viewer.hud.set_enabled(*on);
            true
        }
        ViewerCmd::LoadIbl(path) => {
            match viewer.load_ibl(path) {
                Ok(_) => println!("Loaded IBL: {}", path),
                Err(e) => eprintln!("IBL load failed: {}", e),
            }
            true
        }
        ViewerCmd::IblToggle(on) => {
            viewer.lit_use_ibl = *on;
            if *on && viewer.ibl_renderer.is_none() {
                println!("IBL enabled (no environment loaded; use :ibl load <path> to load HDR)");
            } else if !on {
                println!("IBL disabled");
            }
            viewer.update_lit_uniform();
            true
        }
        ViewerCmd::IblIntensity(v) => {
            viewer.lit_ibl_intensity = (*v).max(0.0);
            viewer.lit_use_ibl = viewer.lit_ibl_intensity > 0.0;
            viewer.update_lit_uniform();
            println!("IBL intensity: {:.2}", viewer.lit_ibl_intensity);
            true
        }
        ViewerCmd::IblRotate(deg) => {
            viewer.lit_ibl_rotation_deg = *deg;
            println!("IBL rotation: {:.1}°", deg);
            true
        }
        ViewerCmd::IblCache(dir) => {
            if let Some(cache_path) = dir {
                viewer.ibl_cache_dir = Some(PathBuf::from(cache_path));
                println!(
                    "IBL cache directory: {} (will be used on next load)",
                    cache_path
                );
                if let Some(ref mut ibl) = viewer.ibl_renderer {
                    let hdr_path = viewer
                        .ibl_hdr_path
                        .as_ref()
                        .map(Path::new)
                        .unwrap_or_else(|| Path::new(""));
                    if let Err(e) = ibl.configure_cache(cache_path, hdr_path) {
                        eprintln!("Failed to configure IBL cache: {}", e);
                    } else {
                        println!("IBL cache reconfigured");
                    }
                }
            } else {
                viewer.ibl_cache_dir = None;
                println!("IBL cache directory cleared (cache will be disabled on next load)");
            }
            true
        }
        ViewerCmd::IblRes(res) => {
            viewer.ibl_base_resolution = Some(*res);
            println!("IBL base resolution: {} (will be used on next load)", res);
            if let Some(ref mut ibl) = viewer.ibl_renderer {
                ibl.set_base_resolution(*res);
                if let Err(e) = ibl.initialize(&viewer.device, &viewer.queue) {
                    eprintln!("Failed to reinitialize IBL with new resolution: {}", e);
                } else {
                    println!("IBL reinitialized with resolution {}", res);
                }
            }
            true
        }
        _ => false,
    }
}

fn normalize_snapshot_path(path: Option<String>) -> String {
    let resolved = path.unwrap_or_else(|| "snapshot.png".to_string());
    let has_sep = resolved.contains('/') || resolved.contains('\\');
    if !has_sep && resolved.starts_with("p5_") {
        let filename = if resolved.ends_with(".png") {
            resolved
        } else {
            format!("{}.png", resolved)
        };
        return PathBuf::from("reports")
            .join("p5")
            .join(filename)
            .to_string_lossy()
            .to_string();
    }
    resolved
}
