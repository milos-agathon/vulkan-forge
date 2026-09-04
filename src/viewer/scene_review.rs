use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::labels::{LabelStyle, LineLabelPlacement};
use crate::viewer::event_loop::update_scene_review_state;
use crate::viewer::terrain::vector_overlay::{
    OverlayPrimitive, VectorOverlayLayer, VectorSourceVertex,
};
use crate::viewer::viewer_enums::ViewerTerrainScatterBatchConfig;
use crate::viewer::viewer_enums::{
    ViewerDenoiseConfig, ViewerDensityVolumeConfig, ViewerDofConfig, ViewerHeightAoConfig,
    ViewerLensEffectsConfig, ViewerMaterialLayerConfig, ViewerMotionBlurConfig, ViewerSkyConfig,
    ViewerSunVisConfig, ViewerTonemapConfig, ViewerVectorOverlayConfig, ViewerVectorVertex,
    ViewerVolumetricsConfig,
};
use crate::viewer::Viewer;

#[derive(Debug, Clone, Default)]
pub struct ViewerRasterOverlayConfig {
    pub name: String,
    pub path: String,
    pub extent: Option<[f32; 4]>,
    pub opacity: Option<f32>,
    pub z_order: Option<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerVectorOverlayLayerConfig {
    pub name: String,
    pub vertices: Vec<ViewerVectorVertex>,
    pub indices: Vec<u32>,
    pub primitive: String,
    pub drape: bool,
    pub drape_offset: f32,
    pub opacity: f32,
    pub depth_bias: f32,
    pub line_width: f32,
    pub point_size: f32,
    pub z_order: i32,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerSceneBaseStateConfig {
    pub preset: Option<Map<String, Value>>,
    pub raster_overlays: Vec<ViewerRasterOverlayConfig>,
    pub vector_overlays: Vec<ViewerVectorOverlayLayerConfig>,
    pub labels: Vec<Map<String, Value>>,
    pub scatter_batches: Vec<ViewerTerrainScatterBatchConfig>,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerReviewLayerConfig {
    pub id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub raster_overlays: Vec<ViewerRasterOverlayConfig>,
    pub vector_overlays: Vec<ViewerVectorOverlayLayerConfig>,
    pub labels: Vec<Map<String, Value>>,
    pub scatter_batches: Vec<ViewerTerrainScatterBatchConfig>,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerSceneVariantConfig {
    pub id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub active_layer_ids: Vec<String>,
    pub preset: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerSceneReviewStateConfig {
    pub base_state: ViewerSceneBaseStateConfig,
    pub review_layers: Vec<ViewerReviewLayerConfig>,
    pub variants: Vec<ViewerSceneVariantConfig>,
    pub active_variant_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SceneVariantSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub active_layer_ids: Vec<String>,
    pub has_preset: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReviewLayerSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub raster_overlay_count: usize,
    pub vector_overlay_count: usize,
    pub label_count: usize,
    pub scatter_batch_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SceneReviewSnapshot {
    pub scene_variants: Vec<SceneVariantSummary>,
    pub review_layers: Vec<ReviewLayerSummary>,
    pub active_scene_variant: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerSceneReviewRegistry {
    pub state: ViewerSceneReviewStateConfig,
    pub layer_visibility_overrides: HashMap<String, bool>,
    pub managed_raster_overlay_ids: Vec<u32>,
    pub managed_vector_overlay_ids: Vec<u32>,
    pub managed_label_ids: Vec<u64>,
}

impl ViewerSceneReviewStateConfig {
    pub fn validate(&self) -> Result<(), String> {
        let mut layer_ids = HashSet::new();
        for layer in &self.review_layers {
            if layer.id.is_empty() {
                return Err("review layer id must be non-empty".to_string());
            }
            if !layer_ids.insert(layer.id.clone()) {
                return Err(format!("Duplicate review layer ID: {}", layer.id));
            }
        }

        let mut variant_ids = HashSet::new();
        for variant in &self.variants {
            if variant.id.is_empty() {
                return Err("scene variant id must be non-empty".to_string());
            }
            if !variant_ids.insert(variant.id.clone()) {
                return Err(format!("Duplicate scene variant ID: {}", variant.id));
            }
            for layer_id in &variant.active_layer_ids {
                if !layer_ids.contains(layer_id) {
                    return Err(format!(
                        "Variant '{}' references unknown review layer ID '{}'",
                        variant.id, layer_id
                    ));
                }
            }
        }

        if let Some(active_variant_id) = &self.active_variant_id {
            if !variant_ids.contains(active_variant_id) {
                return Err(format!(
                    "active_variant_id '{}' does not match any variant",
                    active_variant_id
                ));
            }
        }

        Ok(())
    }

    pub fn variant_summaries(&self) -> Vec<SceneVariantSummary> {
        self.variants
            .iter()
            .map(|variant| SceneVariantSummary {
                id: variant.id.clone(),
                name: variant.name.clone(),
                description: variant.description.clone(),
                active_layer_ids: variant.active_layer_ids.clone(),
                has_preset: variant.preset.is_some(),
            })
            .collect()
    }

    pub fn review_layer_summaries(&self) -> Vec<ReviewLayerSummary> {
        self.review_layers
            .iter()
            .map(|layer| ReviewLayerSummary {
                id: layer.id.clone(),
                name: layer.name.clone(),
                description: layer.description.clone(),
                raster_overlay_count: layer.raster_overlays.len(),
                vector_overlay_count: layer.vector_overlays.len(),
                label_count: layer.labels.len(),
                scatter_batch_count: layer.scatter_batches.len(),
            })
            .collect()
    }

    pub fn snapshot(&self) -> SceneReviewSnapshot {
        SceneReviewSnapshot {
            scene_variants: self.variant_summaries(),
            review_layers: self.review_layer_summaries(),
            active_scene_variant: self.active_variant_id.clone(),
        }
    }

    pub fn layer_by_id(&self, layer_id: &str) -> Option<&ViewerReviewLayerConfig> {
        self.review_layers.iter().find(|layer| layer.id == layer_id)
    }

    pub fn variant_by_id(&self, variant_id: &str) -> Option<&ViewerSceneVariantConfig> {
        self.variants
            .iter()
            .find(|variant| variant.id == variant_id)
    }
}

impl ViewerSceneReviewRegistry {
    pub fn install(&mut self, state: ViewerSceneReviewStateConfig) -> Result<(), String> {
        state.validate()?;
        self.state = state;
        self.layer_visibility_overrides.clear();
        Ok(())
    }

    pub fn apply_variant(&mut self, variant_id: &str) -> Result<(), String> {
        if self.state.variant_by_id(variant_id).is_none() {
            return Err(format!("Unknown scene variant: {variant_id}"));
        }
        self.state.active_variant_id = Some(variant_id.to_string());
        self.layer_visibility_overrides.clear();
        Ok(())
    }

    pub fn set_review_layer_visible(
        &mut self,
        layer_id: &str,
        visible: bool,
    ) -> Result<(), String> {
        if self.state.layer_by_id(layer_id).is_none() {
            return Err(format!("Unknown review layer: {layer_id}"));
        }
        let default_visible = self.default_visible_layer_ids().contains(layer_id);
        if visible == default_visible {
            self.layer_visibility_overrides.remove(layer_id);
        } else {
            self.layer_visibility_overrides
                .insert(layer_id.to_string(), visible);
        }
        Ok(())
    }

    pub fn snapshot(&self) -> SceneReviewSnapshot {
        let mut snapshot = self.state.snapshot();
        snapshot.active_scene_variant = self.state.active_variant_id.clone();
        snapshot
    }

    pub fn effective_state(&self) -> Result<ViewerSceneBaseStateConfig, String> {
        self.state.validate()?;
        let mut result = self.state.base_state.clone();
        let visible_layer_ids = self.visible_layer_ids();
        for layer in &self.state.review_layers {
            if visible_layer_ids.contains(&layer.id) {
                result.raster_overlays.extend(layer.raster_overlays.clone());
                result.vector_overlays.extend(layer.vector_overlays.clone());
                result.labels.extend(layer.labels.clone());
                result.scatter_batches.extend(layer.scatter_batches.clone());
            }
        }
        if let Some(active_variant_id) = &self.state.active_variant_id {
            if let Some(variant) = self.state.variant_by_id(active_variant_id) {
                if let Some(preset) = &variant.preset {
                    result.preset = Some(preset.clone());
                }
            }
        }
        Ok(result)
    }

    fn default_visible_layer_ids(&self) -> HashSet<String> {
        self.state
            .active_variant_id
            .as_deref()
            .and_then(|variant_id| self.state.variant_by_id(variant_id))
            .map(|variant| variant.active_layer_ids.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn visible_layer_ids(&self) -> HashSet<String> {
        let mut visible = self.default_visible_layer_ids();
        for (layer_id, override_visible) in &self.layer_visibility_overrides {
            if *override_visible {
                visible.insert(layer_id.clone());
            } else {
                visible.remove(layer_id);
            }
        }
        visible
    }
}

impl Viewer {
    pub(crate) fn set_scene_review_state(
        &mut self,
        state: ViewerSceneReviewStateConfig,
    ) -> Result<(), String> {
        let previous = self.scene_review_registry.clone();
        self.scene_review_registry.install(state)?;
        if let Err(err) = self.reapply_scene_review_state() {
            self.scene_review_registry = previous;
            return Err(err);
        }
        update_scene_review_state(self.scene_review_registry.snapshot());
        Ok(())
    }

    pub(crate) fn apply_scene_variant(&mut self, variant_id: &str) -> Result<(), String> {
        let previous = self.scene_review_registry.clone();
        self.scene_review_registry.apply_variant(variant_id)?;
        if let Err(err) = self.reapply_scene_review_state() {
            self.scene_review_registry = previous;
            return Err(err);
        }
        update_scene_review_state(self.scene_review_registry.snapshot());
        Ok(())
    }

    pub(crate) fn set_review_layer_visible(
        &mut self,
        layer_id: &str,
        visible: bool,
    ) -> Result<(), String> {
        let previous = self.scene_review_registry.clone();
        self.scene_review_registry
            .set_review_layer_visible(layer_id, visible)?;
        if let Err(err) = self.reapply_scene_review_state() {
            self.scene_review_registry = previous;
            return Err(err);
        }
        update_scene_review_state(self.scene_review_registry.snapshot());
        Ok(())
    }

    pub(crate) fn reapply_scene_review_state(&mut self) -> Result<(), String> {
        let effective = self.scene_review_registry.effective_state()?;
        let content_anchor = self.validate_scene_review_effective(&effective)?;
        // All fallible parsing happens before any live registry/runtime change.
        let staged_labels = effective
            .labels
            .iter()
            .map(|payload| stage_review_label(payload, &content_anchor))
            .collect::<Result<Vec<_>, _>>()?;
        scene_review_injected_failure("after_label_staging")?;
        let staged_vector_layers = effective
            .vector_overlays
            .iter()
            .map(review_vector_layer)
            .collect::<Vec<_>>();

        let old_raster_ids = self
            .scene_review_registry
            .managed_raster_overlay_ids
            .clone();
        let old_vector_ids = self
            .scene_review_registry
            .managed_vector_overlay_ids
            .clone();
        let old_label_ids = self.scene_review_registry.managed_label_ids.clone();

        // Detached candidates preserve unmanaged layers. Their allocations and
        // next-ID counters are unreachable from the live renderer until swap.
        let (staged_stacks, raster_ids, vector_ids, staged_bvhs) =
            if let Some(terrain_viewer) = self.terrain_viewer.as_ref() {
                let (raster_stack, raster_ids) = terrain_viewer
                    .stage_review_raster_replacement(&old_raster_ids, &effective.raster_overlays)
                    .map_err(|error| format!("review raster staging failed: {error}"))?;
                scene_review_injected_failure("after_raster_composite")?;
                let (vector_stack, vector_ids) = terrain_viewer
                    .stage_review_vector_replacement(
                        &old_vector_ids,
                        staged_vector_layers,
                        &content_anchor,
                    )
                    .map_err(|error| format!("review vector staging failed: {error}"))?;
                scene_review_injected_failure("after_vector_allocation")?;
                let staged_bvhs = vector_stack
                    .render_layer_data()
                    .filter(|(id, ..)| vector_ids.contains(id))
                    .filter_map(|(id, name, vertices, indices, primitive)| {
                        crate::viewer::terrain::vector_overlay::build_layer_bvh(
                            id, name, vertices, indices, primitive,
                        )
                    })
                    .collect::<Vec<_>>();
                (
                    Some((raster_stack, vector_stack)),
                    raster_ids,
                    vector_ids,
                    staged_bvhs,
                )
            } else {
                (None, Vec::new(), Vec::new(), Vec::new())
            };

        #[cfg(feature = "enable-gpu-instancing")]
        let staged_scatter_batches = self
            .terrain_viewer
            .as_ref()
            .map(|terrain_viewer| {
                terrain_viewer.stage_scatter_batches_from_configs(&effective.scatter_batches)
            })
            .transpose()
            .map_err(|err| format!("Failed to stage review scatter batches: {err:#}"))?;
        #[cfg(not(feature = "enable-gpu-instancing"))]
        {
            if !effective.scatter_batches.is_empty() {
                return Err(
                    "Review scatter batches require Cargo feature 'enable-gpu-instancing'"
                        .to_string(),
                );
            }
        }

        // The PBR setter stages its fallible shadow replacement transactionally.
        // Run it only after every other candidate is ready, but before any
        // overlay, BVH, label, camera, sky, or registry state is committed.
        self.apply_scene_review_pbr_preset(effective.preset.as_ref())?;

        // Commit: no fallible work remains. Drop old managed BVHs, atomically
        // swap both complete stacks, then update labels and registry last.
        for id in &old_vector_ids {
            self.unified_picking.remove_layer_bvh(*id);
        }
        if let (Some(terrain_viewer), Some((raster_stack, vector_stack))) =
            (self.terrain_viewer.as_mut(), staged_stacks)
        {
            terrain_viewer.commit_review_stacks(raster_stack, vector_stack);
        }
        #[cfg(feature = "enable-gpu-instancing")]
        if let (Some(terrain_viewer), Some(scatter_batches)) =
            (self.terrain_viewer.as_mut(), staged_scatter_batches)
        {
            terrain_viewer.commit_scatter_batches(scatter_batches);
        }
        for bvh in staged_bvhs {
            self.unified_picking.register_layer_bvh(bvh);
        }
        for id in old_label_ids {
            let _ = self.remove_label(id);
        }
        let label_ids = staged_labels
            .into_iter()
            .map(|label| label.commit(self))
            .collect::<Vec<_>>();
        self.apply_scene_review_preset(effective.preset.as_ref());

        self.scene_review_registry.managed_raster_overlay_ids = raster_ids;
        self.scene_review_registry.managed_vector_overlay_ids = vector_ids;
        self.scene_review_registry.managed_label_ids = label_ids;

        Ok(())
    }

    fn validate_scene_review_effective(
        &self,
        effective: &ViewerSceneBaseStateConfig,
    ) -> Result<crate::camera::Anchor, String> {
        let mut anchor = self.prospective_frame_camera().anchor;
        if let (Some(terrain), Some(preset)) = (
            self.terrain_viewer
                .as_ref()
                .and_then(|scene| scene.terrain.as_ref()),
            effective.preset.as_ref(),
        ) {
            let camera = preset.get("camera").and_then(Value::as_object);
            let phi = first_f32(preset, &["cam_phi", "phi_deg", "phi"])
                .or_else(|| camera.and_then(|map| first_f32(map, &["phi_deg", "phi"])))
                .unwrap_or(terrain.cam_phi_deg);
            let theta = first_f32(preset, &["cam_theta", "theta_deg", "theta"])
                .or_else(|| camera.and_then(|map| first_f32(map, &["theta_deg", "theta"])))
                .unwrap_or(terrain.cam_theta_deg);
            let radius = first_f32(preset, &["cam_radius", "radius"])
                .or_else(|| camera.and_then(|map| first_f32(map, &["radius", "distance"])))
                .unwrap_or(terrain.cam_radius);
            let fov = first_f32(preset, &["cam_fov", "fov_deg", "fov"])
                .or_else(|| camera.and_then(|map| first_f32(map, &["fov_deg", "fov"])))
                .unwrap_or(terrain.cam_fov_deg);
            terrain
                .validate_camera_state(
                    &self.camera_anchor,
                    phi,
                    theta,
                    radius,
                    fov,
                    terrain.cam_target,
                )
                .map_err(|err| err.to_string())?;
            anchor = self.camera_anchor;
            anchor.rebase_if_needed(DVec3::new(terrain.cam_target.x, 0.0, terrain.cam_target.z));
        }

        for overlay in &effective.vector_overlays {
            for vertex in &overlay.vertices {
                crate::viewer::camera_controller::validate_world_point(
                    crate::viewer::camera_controller::CoordRole::Content,
                    DVec3::from(vertex.position),
                    &anchor,
                )
                .map_err(|err| err.to_string())?;
            }
        }
        for payload in &effective.labels {
            for point in review_label_world_points(payload)? {
                crate::viewer::camera_controller::validate_world_point(
                    crate::viewer::camera_controller::CoordRole::Content,
                    point,
                    &anchor,
                )
                .map_err(|err| err.to_string())?;
            }
        }
        Ok(anchor)
    }
}

fn review_vector_layer(overlay: &ViewerVectorOverlayLayerConfig) -> VectorOverlayLayer {
    let primitive =
        OverlayPrimitive::from_str(&overlay.primitive).unwrap_or(OverlayPrimitive::Triangles);
    VectorOverlayLayer {
        name: overlay.name.clone(),
        source_vertices: overlay
            .vertices
            .iter()
            .map(|vertex| VectorSourceVertex::from(*vertex))
            .collect(),
        vertices: Vec::new(),
        indices: overlay.indices.clone(),
        primitive,
        drape: overlay.drape,
        drape_offset: overlay.drape_offset,
        opacity: overlay.opacity,
        depth_bias: overlay.depth_bias,
        line_width: overlay.line_width,
        point_size: overlay.point_size,
        visible: true,
        z_order: overlay.z_order,
    }
}

impl Viewer {
    fn apply_scene_review_pbr_preset(
        &mut self,
        preset: Option<&Map<String, Value>>,
    ) -> Result<(), String> {
        let Some(preset) = preset else {
            return Ok(());
        };

        let enabled = first_bool(preset, &["enabled", "terrain_pbr_enabled"]);
        let hdr_path = first_string(preset, &["hdr_path", "hdr"]);
        let ibl_intensity = first_f32(preset, &["ibl_intensity"]);
        let hdr_rotate_deg = first_f32(preset, &["hdr_rotate_deg", "hdr_rotate"]);
        let shadow_technique = first_string(preset, &["shadow_technique"]);
        let shadow_map_res = first_u32(preset, &["shadow_map_res"]);
        let exposure = first_f32(preset, &["exposure"]);
        let msaa = first_u32(preset, &["msaa"]);
        let normal_strength = first_f32(preset, &["normal_strength"]);
        let height_ao = preset.get("height_ao").and_then(parse_height_ao_config);
        let sun_visibility = preset
            .get("sun_visibility")
            .and_then(parse_sun_visibility_config);
        let materials = preset.get("materials").and_then(parse_materials_config);
        let vector_overlay = preset
            .get("vector_overlay")
            .and_then(parse_vector_overlay_config);
        let tonemap = preset.get("tonemap").and_then(parse_tonemap_config);
        let dof = preset.get("dof").and_then(parse_dof_config);
        let motion_blur = preset.get("motion_blur").and_then(parse_motion_blur_config);
        let lens_effects = preset
            .get("lens_effects")
            .and_then(parse_lens_effects_config);
        let denoise = preset.get("denoise").and_then(parse_denoise_config);
        let volumetrics = preset.get("volumetrics").and_then(parse_volumetrics_config);
        let debug_mode = first_u32(preset, &["debug_mode"]);

        if let Some(terrain_viewer) = self.terrain_viewer.as_mut() {
            terrain_viewer
                .set_terrain_pbr(
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
                    lens_effects,
                    dof,
                    motion_blur,
                    volumetrics,
                    denoise,
                    debug_mode,
                )
                .map_err(|error| format!("scene-review terrain PBR preset rejected: {error}"))?;
        }
        Ok(())
    }

    fn apply_scene_review_preset(&mut self, preset: Option<&Map<String, Value>>) {
        let Some(preset) = preset else {
            return;
        };

        let mut phi_deg = first_f32(preset, &["cam_phi", "phi_deg", "phi"]);
        let mut theta_deg = first_f32(preset, &["cam_theta", "theta_deg", "theta"]);
        let mut radius = first_f32(preset, &["cam_radius", "radius"]);
        let mut fov_deg = first_f32(preset, &["cam_fov", "fov_deg", "fov"]);
        if let Some(camera) = preset.get("camera").and_then(Value::as_object) {
            phi_deg = phi_deg.or(first_f32(camera, &["phi_deg", "phi"]));
            theta_deg = theta_deg.or(first_f32(camera, &["theta_deg", "theta"]));
            radius = radius.or(first_f32(camera, &["radius", "distance"]));
            fov_deg = fov_deg.or(first_f32(camera, &["fov_deg", "fov"]));
        }

        let mut sun_azimuth = first_f32(preset, &["sun_azimuth", "sun_azimuth_deg"]);
        let mut sun_elevation = first_f32(preset, &["sun_elevation", "sun_elevation_deg"]);
        let mut sun_intensity = first_f32(preset, &["sun_intensity"]);
        if let Some(sun) = preset.get("sun").and_then(Value::as_object) {
            sun_azimuth = sun_azimuth.or(first_f32(sun, &["azimuth_deg", "azimuth"]));
            sun_elevation = sun_elevation.or(first_f32(sun, &["elevation_deg", "elevation"]));
            sun_intensity = sun_intensity.or(first_f32(sun, &["intensity"]));
        }

        let ambient = first_f32(preset, &["ambient"]);
        let zscale = first_f32(preset, &["z_scale", "zscale"]);
        let shadow = first_f32(preset, &["shadow", "shadow_intensity"]);
        let background = first_array3(preset, &["background", "background_color"]);
        let water_level = first_f32(preset, &["water_level"]);
        let water_color = first_array3(preset, &["water_color"]);

        let sky = preset.get("sky").and_then(parse_sky_config);

        if let Some(ref mut terrain_viewer) = self.terrain_viewer {
            if let Some(ref mut terrain) = terrain_viewer.terrain {
                if let Some(phi) = phi_deg {
                    terrain.cam_phi_deg = phi;
                }
                if let Some(theta) = theta_deg {
                    terrain.cam_theta_deg = theta.clamp(5.0, 85.0);
                }
                if let Some(r) = radius {
                    terrain.cam_radius = r.clamp(100.0, 50_000.0);
                }
                if let Some(fov) = fov_deg {
                    terrain.cam_fov_deg = fov.clamp(10.0, 120.0);
                }
                if let Some(azimuth) = sun_azimuth {
                    terrain.sun_azimuth_deg = azimuth;
                }
                if let Some(elevation) = sun_elevation {
                    terrain.sun_elevation_deg = elevation.clamp(-90.0, 90.0);
                }
                if let Some(intensity) = sun_intensity {
                    terrain.sun_intensity = intensity.max(0.0);
                }
                if let Some(value) = ambient {
                    terrain.ambient = value.clamp(0.0, 1.0);
                }
                if let Some(value) = zscale {
                    terrain.z_scale = value.max(0.01);
                }
                if let Some(value) = shadow {
                    terrain.shadow_intensity = value.clamp(0.0, 1.0);
                }
                if let Some(value) = background {
                    terrain.background_color = value;
                }
                if let Some(value) = water_level {
                    terrain.water_level = value;
                }
                if let Some(value) = water_color {
                    terrain.water_color = value;
                }
            }
        }

        if let Some(cfg) = sky {
            self.sky_enabled = cfg.enabled;
            self.sky_turbidity = cfg.turbidity;
            self.sky_ground_albedo = cfg.ground_albedo;
            self.sky_sun_intensity = cfg.sun_intensity;
            self.sky_exposure = cfg.sky_exposure;
        }
    }
}

fn review_label_world_points(payload: &Map<String, Value>) -> Result<Vec<DVec3>, String> {
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "Review label payload missing 'kind'".to_string())?;
    let value = Value::Object(payload.clone());
    match kind {
        "point" => serde_json::from_value::<PointLabelPayload>(value)
            .map(|item| vec![DVec3::from(item.world_pos)])
            .map_err(|err| err.to_string()),
        "line" => serde_json::from_value::<LineLabelPayload>(value)
            .map(|item| item.polyline.into_iter().map(DVec3::from).collect())
            .map_err(|err| err.to_string()),
        "curved" => serde_json::from_value::<CurvedLabelPayload>(value)
            .map(|item| item.polyline.into_iter().map(DVec3::from).collect())
            .map_err(|err| err.to_string()),
        "callout" => serde_json::from_value::<CalloutLabelPayload>(value)
            .map(|item| vec![DVec3::from(item.anchor)])
            .map_err(|err| err.to_string()),
        other => Err(format!("Unsupported review label kind: {other}")),
    }
}

enum StagedReviewLabel {
    Point {
        text: String,
        world_pos: DVec3,
        style: LabelStyle,
    },
    Line {
        text: String,
        polyline: Vec<DVec3>,
        style: LabelStyle,
        placement: LineLabelPlacement,
        repeat_distance: f32,
    },
}

impl StagedReviewLabel {
    fn commit(self, viewer: &mut Viewer) -> u64 {
        match self {
            Self::Point {
                text,
                world_pos,
                style,
            } => viewer.add_label(&text, (world_pos.x, world_pos.y, world_pos.z), Some(style)),
            Self::Line {
                text,
                polyline,
                style,
                placement,
                repeat_distance,
            } => {
                viewer
                    .label_manager
                    .add_line_label(text, polyline, style, placement, repeat_distance)
                    .0
            }
        }
    }
}

fn stage_review_label(
    payload: &Map<String, Value>,
    anchor: &crate::camera::Anchor,
) -> Result<StagedReviewLabel, String> {
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "Review label payload missing 'kind'".to_string())?;
    let payload_value = Value::Object(payload.clone());
    match kind {
        "point" => {
            let point: PointLabelPayload =
                serde_json::from_value(payload_value).map_err(|e| e.to_string())?;
            let mut style = LabelStyle::default();
            apply_point_style(&mut style, &point);
            crate::viewer::camera_controller::validate_world_point(
                crate::viewer::camera_controller::CoordRole::Content,
                DVec3::from(point.world_pos),
                anchor,
            )
            .map_err(|err| err.to_string())?;
            Ok(StagedReviewLabel::Point {
                text: point.text,
                world_pos: DVec3::from(point.world_pos),
                style,
            })
        }
        "line" => {
            let line: LineLabelPayload =
                serde_json::from_value(payload_value).map_err(|e| e.to_string())?;
            let mut style = LabelStyle::default();
            apply_line_style(&mut style, &line);
            let placement = match line.placement.as_deref() {
                Some("along") => LineLabelPlacement::Along,
                _ => LineLabelPlacement::Center,
            };
            let polyline: Vec<DVec3> = line.polyline.iter().copied().map(DVec3::from).collect();
            for point in polyline.iter().copied() {
                crate::viewer::camera_controller::validate_world_point(
                    crate::viewer::camera_controller::CoordRole::Content,
                    point,
                    anchor,
                )
                .map_err(|err| err.to_string())?;
            }
            Ok(StagedReviewLabel::Line {
                text: line.text,
                polyline,
                style,
                placement,
                repeat_distance: line.repeat_distance.unwrap_or(0.0),
            })
        }
        "curved" => {
            let curved: CurvedLabelPayload =
                serde_json::from_value(payload_value).map_err(|e| e.to_string())?;
            let mut style = LabelStyle::default();
            apply_curved_style(&mut style, &curved);
            let polyline: Vec<DVec3> = curved.polyline.iter().copied().map(DVec3::from).collect();
            for point in polyline.iter().copied() {
                crate::viewer::camera_controller::validate_world_point(
                    crate::viewer::camera_controller::CoordRole::Content,
                    point,
                    anchor,
                )
                .map_err(|err| err.to_string())?;
            }
            Ok(StagedReviewLabel::Line {
                text: curved.text,
                polyline,
                style,
                placement: LineLabelPlacement::Along,
                repeat_distance: 0.0,
            })
        }
        "callout" => {
            let callout: CalloutLabelPayload =
                serde_json::from_value(payload_value).map_err(|e| e.to_string())?;
            let mut style = LabelStyle::default();
            if let Some(size) = callout.text_size {
                style.size = size;
            }
            if let Some(color) = callout.text_color {
                style.color = color;
            }
            if let Some(offset) = callout.offset {
                style.offset = offset;
                style.flags.leader = true;
            }
            crate::viewer::camera_controller::validate_world_point(
                crate::viewer::camera_controller::CoordRole::Content,
                DVec3::from(callout.anchor),
                anchor,
            )
            .map_err(|err| err.to_string())?;
            Ok(StagedReviewLabel::Point {
                text: callout.text,
                world_pos: DVec3::from(callout.anchor),
                style,
            })
        }
        other => Err(format!("Unsupported review label kind: {other}")),
    }
}

#[cfg(test)]
static SCENE_REVIEW_FAILURE_POINT: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
fn scene_review_injected_failure(point: &str) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    let expected = match point {
        "after_label_staging" => 1,
        "after_vector_allocation" => 2,
        "after_raster_composite" => 3,
        _ => 0,
    };
    if SCENE_REVIEW_FAILURE_POINT.load(Ordering::SeqCst) == expected {
        Err(format!("injected scene-review failure: {point}"))
    } else {
        Ok(())
    }
}

#[cfg(not(test))]
fn scene_review_injected_failure(point: &str) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};

    static MATCHING_CALLS: AtomicU64 = AtomicU64::new(0);
    if std::env::var("RUN_M06_VIEWER_CI").as_deref() != Ok("1") {
        return Ok(());
    }
    let Ok(spec) = std::env::var("FORGE3D_M06_SCENE_REVIEW_FAILPOINT") else {
        return Ok(());
    };
    let (wanted, nth) = spec
        .rsplit_once(':')
        .and_then(|(wanted, nth)| nth.parse::<u64>().ok().map(|nth| (wanted, nth)))
        .unwrap_or((spec.as_str(), 1));
    if wanted != point || nth == 0 {
        return Ok(());
    }
    let call = MATCHING_CALLS.fetch_add(1, Ordering::SeqCst) + 1;
    if call == nth {
        Err(format!(
            "injected scene-review failure: point={point} matching_call={call}"
        ))
    } else {
        Ok(())
    }
}

fn apply_point_style(style: &mut LabelStyle, payload: &PointLabelPayload) {
    if let Some(size) = payload.size {
        style.size = size;
    }
    if let Some(color) = payload.color {
        style.color = color;
    }
    if let Some(color) = payload.halo_color {
        style.halo_color = color;
    }
    if let Some(width) = payload.halo_width {
        style.halo_width = width;
    }
    if let Some(priority) = payload.priority {
        style.priority = priority;
    }
    if let Some(min_zoom) = payload.min_zoom {
        style.min_zoom = min_zoom;
    }
    if let Some(max_zoom) = payload.max_zoom {
        style.max_zoom = max_zoom;
    }
    if let Some(offset) = payload.offset {
        style.offset = offset;
    }
    if let Some(rotation) = payload.rotation {
        style.rotation = rotation;
    }
    if let Some(underline) = payload.underline {
        style.flags.underline = underline;
    }
    if let Some(small_caps) = payload.small_caps {
        style.flags.small_caps = small_caps;
    }
    if let Some(leader) = payload.leader {
        style.flags.leader = leader;
    }
    if let Some(angle) = payload.horizon_fade_angle {
        style.horizon_fade_angle = angle;
    }
}

fn apply_line_style(style: &mut LabelStyle, payload: &LineLabelPayload) {
    if let Some(size) = payload.size {
        style.size = size;
    }
    if let Some(color) = payload.color {
        style.color = color;
    }
    if let Some(color) = payload.halo_color {
        style.halo_color = color;
    }
    if let Some(width) = payload.halo_width {
        style.halo_width = width;
    }
    if let Some(priority) = payload.priority {
        style.priority = priority;
    }
    if let Some(min_zoom) = payload.min_zoom {
        style.min_zoom = min_zoom;
    }
    if let Some(max_zoom) = payload.max_zoom {
        style.max_zoom = max_zoom;
    }
}

fn apply_curved_style(style: &mut LabelStyle, payload: &CurvedLabelPayload) {
    if let Some(size) = payload.size {
        style.size = size;
    }
    if let Some(color) = payload.color {
        style.color = color;
    }
    if let Some(color) = payload.halo_color {
        style.halo_color = color;
    }
    if let Some(width) = payload.halo_width {
        style.halo_width = width;
    }
    if let Some(priority) = payload.priority {
        style.priority = priority;
    }
}

fn first_f32(map: &Map<String, Value>, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| {
        map.get(*key)
            .and_then(Value::as_f64)
            .map(narrow_nonposition_scalar)
    })
}

fn narrow_nonposition_scalar(value: f64) -> f32 {
    crate::camera::Anchor::direction_to_render(DVec3::new(value, 0.0, 0.0)).x
}

fn first_u32(map: &Map<String, Value>, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| {
        map.get(*key)
            .and_then(Value::as_u64)
            .map(|value| value as u32)
    })
}

fn first_bool(map: &Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_bool))
}

fn first_string(map: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str).map(ToOwned::to_owned))
}

fn first_array3(map: &Map<String, Value>, keys: &[&str]) -> Option<[f32; 3]> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(value_as_array3))
}

fn value_as_array3(value: &Value) -> Option<[f32; 3]> {
    let array = value.as_array()?;
    (array.len() == 3).then_some([
        narrow_nonposition_scalar(array.first()?.as_f64()?),
        narrow_nonposition_scalar(array.get(1)?.as_f64()?),
        narrow_nonposition_scalar(array.get(2)?.as_f64()?),
    ])
}

fn value_as_array4(value: &Value) -> Option<[f32; 4]> {
    let array = value.as_array()?;
    (array.len() == 4).then_some([
        narrow_nonposition_scalar(array.first()?.as_f64()?),
        narrow_nonposition_scalar(array.get(1)?.as_f64()?),
        narrow_nonposition_scalar(array.get(2)?.as_f64()?),
        narrow_nonposition_scalar(array.get(3)?.as_f64()?),
    ])
}

fn parse_height_ao_config(value: &Value) -> Option<ViewerHeightAoConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerHeightAoConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_u32(map, &["directions"]) {
        cfg.directions = v;
    }
    if let Some(v) = first_u32(map, &["steps"]) {
        cfg.steps = v;
    }
    if let Some(v) = first_f32(map, &["max_distance"]) {
        cfg.max_distance = v;
    }
    if let Some(v) = first_f32(map, &["strength"]) {
        cfg.strength = v;
    }
    if let Some(v) = first_f32(map, &["resolution_scale"]) {
        cfg.resolution_scale = v;
    }
    Some(cfg)
}

fn parse_sun_visibility_config(value: &Value) -> Option<ViewerSunVisConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerSunVisConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_string(map, &["mode"]) {
        cfg.mode = v;
    }
    if let Some(v) = first_u32(map, &["samples"]) {
        cfg.samples = v;
    }
    if let Some(v) = first_u32(map, &["steps"]) {
        cfg.steps = v;
    }
    if let Some(v) = first_f32(map, &["max_distance"]) {
        cfg.max_distance = v;
    }
    if let Some(v) = first_f32(map, &["softness"]) {
        cfg.softness = v;
    }
    if let Some(v) = first_f32(map, &["bias"]) {
        cfg.bias = v;
    }
    if let Some(v) = first_f32(map, &["resolution_scale"]) {
        cfg.resolution_scale = v;
    }
    Some(cfg)
}

fn parse_materials_config(value: &Value) -> Option<ViewerMaterialLayerConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerMaterialLayerConfig::default();
    if let Some(v) = first_bool(map, &["snow_enabled"]) {
        cfg.snow_enabled = v;
    }
    if let Some(v) = first_f32(map, &["snow_altitude_min"]) {
        cfg.snow_altitude_min = v;
    }
    if let Some(v) = first_f32(map, &["snow_altitude_blend"]) {
        cfg.snow_altitude_blend = v;
    }
    if let Some(v) = first_f32(map, &["snow_slope_max"]) {
        cfg.snow_slope_max = v;
    }
    if let Some(v) = first_bool(map, &["rock_enabled"]) {
        cfg.rock_enabled = v;
    }
    if let Some(v) = first_f32(map, &["rock_slope_min"]) {
        cfg.rock_slope_min = v;
    }
    if let Some(v) = first_bool(map, &["wetness_enabled"]) {
        cfg.wetness_enabled = v;
    }
    if let Some(v) = first_f32(map, &["wetness_strength"]) {
        cfg.wetness_strength = v;
    }
    Some(cfg)
}

fn parse_vector_overlay_config(value: &Value) -> Option<ViewerVectorOverlayConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerVectorOverlayConfig::default();
    if let Some(v) = first_bool(map, &["depth_test"]) {
        cfg.depth_test = v;
    }
    if let Some(v) = first_f32(map, &["depth_bias"]) {
        cfg.depth_bias = v;
    }
    if let Some(v) = first_bool(map, &["halo_enabled"]) {
        cfg.halo_enabled = v;
    }
    if let Some(v) = first_f32(map, &["halo_width"]) {
        cfg.halo_width = v;
    }
    if let Some(value) = map.get("halo_color").and_then(value_as_array4) {
        cfg.halo_color = value;
    }
    Some(cfg)
}

fn parse_tonemap_config(value: &Value) -> Option<ViewerTonemapConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerTonemapConfig::default();
    if let Some(v) = first_string(map, &["operator"]) {
        cfg.operator = v;
    }
    if let Some(v) = first_f32(map, &["white_point"]) {
        cfg.white_point = v;
    }
    if let Some(v) = first_bool(map, &["white_balance_enabled"]) {
        cfg.white_balance_enabled = v;
    }
    if let Some(v) = first_f32(map, &["temperature"]) {
        cfg.temperature = v;
    }
    if let Some(v) = first_f32(map, &["tint"]) {
        cfg.tint = v;
    }
    Some(cfg)
}

fn parse_dof_config(value: &Value) -> Option<ViewerDofConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerDofConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_f32(map, &["f_stop"]) {
        cfg.f_stop = v;
    }
    if let Some(v) = first_f32(map, &["focus_distance"]) {
        cfg.focus_distance = v;
    }
    if let Some(v) = first_f32(map, &["focal_length"]) {
        cfg.focal_length = v;
    }
    if let Some(v) = first_f32(map, &["tilt_pitch"]) {
        cfg.tilt_pitch = v;
    }
    if let Some(v) = first_f32(map, &["tilt_yaw"]) {
        cfg.tilt_yaw = v;
    }
    if let Some(v) = first_string(map, &["quality"]) {
        cfg.quality = v;
    }
    Some(cfg)
}

fn parse_motion_blur_config(value: &Value) -> Option<ViewerMotionBlurConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerMotionBlurConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_u32(map, &["samples"]) {
        cfg.samples = v;
    }
    if let Some(v) = first_f32(map, &["shutter_open"]) {
        cfg.shutter_open = v;
    }
    if let Some(v) = first_f32(map, &["shutter_close"]) {
        cfg.shutter_close = v;
    }
    if let Some(v) = first_f32(map, &["cam_phi_delta"]) {
        cfg.cam_phi_delta = v;
    }
    if let Some(v) = first_f32(map, &["cam_theta_delta"]) {
        cfg.cam_theta_delta = v;
    }
    if let Some(v) = first_f32(map, &["cam_radius_delta"]) {
        cfg.cam_radius_delta = v;
    }
    Some(cfg)
}

fn parse_lens_effects_config(value: &Value) -> Option<ViewerLensEffectsConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerLensEffectsConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_f32(map, &["distortion"]) {
        cfg.distortion = v;
    }
    if let Some(v) = first_f32(map, &["chromatic_aberration"]) {
        cfg.chromatic_aberration = v;
    }
    if let Some(v) = first_f32(map, &["vignette_strength"]) {
        cfg.vignette_strength = v;
    }
    if let Some(v) = first_f32(map, &["vignette_radius"]) {
        cfg.vignette_radius = v;
    }
    if let Some(v) = first_f32(map, &["vignette_softness"]) {
        cfg.vignette_softness = v;
    }
    Some(cfg)
}

fn parse_denoise_config(value: &Value) -> Option<ViewerDenoiseConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerDenoiseConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_string(map, &["method"]) {
        cfg.method = v;
    }
    if let Some(v) = first_u32(map, &["iterations"]) {
        cfg.iterations = v;
    }
    if let Some(v) = first_f32(map, &["sigma_color"]) {
        cfg.sigma_color = v;
    }
    Some(cfg)
}

fn parse_density_volume_config(value: &Value) -> Option<ViewerDensityVolumeConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerDensityVolumeConfig::default();
    if let Some(v) = first_string(map, &["preset"]) {
        cfg.preset = v;
    }
    if let Some(v) = map.get("center").and_then(value_as_array3) {
        cfg.center = v;
    }
    if let Some(v) = map.get("size").and_then(value_as_array3) {
        cfg.size = v;
    }
    if let Some(v) = map.get("resolution").and_then(Value::as_array) {
        if v.len() == 3 {
            let parsed = [
                v[0].as_u64().unwrap_or(cfg.resolution[0] as u64) as u32,
                v[1].as_u64().unwrap_or(cfg.resolution[1] as u64) as u32,
                v[2].as_u64().unwrap_or(cfg.resolution[2] as u64) as u32,
            ];
            cfg.resolution = parsed;
        }
    }
    if let Some(v) = first_f32(map, &["density_scale"]) {
        cfg.density_scale = v;
    }
    if let Some(v) = first_f32(map, &["edge_softness"]) {
        cfg.edge_softness = v;
    }
    if let Some(v) = first_f32(map, &["noise_strength"]) {
        cfg.noise_strength = v;
    }
    if let Some(v) = first_f32(map, &["floor_offset"]) {
        cfg.floor_offset = v;
    }
    if let Some(v) = first_f32(map, &["ceiling"]) {
        cfg.ceiling = v;
    }
    if let Some(v) = first_f32(map, &["plume_spread"]) {
        cfg.plume_spread = v;
    }
    if let Some(v) = map.get("wind").and_then(value_as_array3) {
        cfg.wind = v;
    }
    if let Some(v) = first_u32(map, &["seed"]) {
        cfg.seed = v;
    }
    Some(cfg)
}

fn parse_volumetrics_config(value: &Value) -> Option<ViewerVolumetricsConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerVolumetricsConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_string(map, &["mode"]) {
        cfg.mode = v;
    }
    if let Some(v) = first_f32(map, &["density"]) {
        cfg.density = v;
    }
    if let Some(v) = first_f32(map, &["height_falloff"]) {
        cfg.height_falloff = v;
    }
    if let Some(v) = first_f32(map, &["scattering"]) {
        cfg.scattering = v;
    }
    if let Some(v) = first_f32(map, &["absorption"]) {
        cfg.absorption = v;
    }
    if let Some(v) = first_bool(map, &["light_shafts"]) {
        cfg.light_shafts = v;
    }
    if let Some(v) = first_f32(map, &["shaft_intensity"]) {
        cfg.shaft_intensity = v;
    }
    if let Some(v) = first_u32(map, &["steps"]) {
        cfg.steps = v;
    }
    if let Some(v) = first_bool(map, &["half_res"]) {
        cfg.half_res = v;
    }
    cfg.density_volumes = map
        .get("density_volumes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(parse_density_volume_config)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(cfg)
}

fn parse_sky_config(value: &Value) -> Option<ViewerSkyConfig> {
    let map = value.as_object()?;
    let mut cfg = ViewerSkyConfig::default();
    if let Some(v) = first_bool(map, &["enabled"]) {
        cfg.enabled = v;
    }
    if let Some(v) = first_f32(map, &["turbidity"]) {
        cfg.turbidity = v;
    }
    if let Some(v) = first_f32(map, &["ground_albedo"]) {
        cfg.ground_albedo = v;
    }
    if let Some(v) = first_f32(map, &["sun_intensity"]) {
        cfg.sun_intensity = v;
    }
    if let Some(v) = first_bool(map, &["aerial_perspective"]) {
        cfg.aerial_perspective = v;
    }
    if let Some(v) = first_f32(map, &["sky_exposure", "exposure"]) {
        cfg.sky_exposure = v;
    }
    Some(cfg)
}

#[derive(Debug, Clone, Deserialize)]
struct PointLabelPayload {
    text: String,
    world_pos: [f64; 3],
    #[serde(default)]
    size: Option<f32>,
    #[serde(default)]
    color: Option<[f32; 4]>,
    #[serde(default)]
    halo_color: Option<[f32; 4]>,
    #[serde(default)]
    halo_width: Option<f32>,
    #[serde(default)]
    priority: Option<i32>,
    #[serde(default)]
    min_zoom: Option<f32>,
    #[serde(default)]
    max_zoom: Option<f32>,
    #[serde(default)]
    offset: Option<[f32; 2]>,
    #[serde(default)]
    rotation: Option<f32>,
    #[serde(default)]
    underline: Option<bool>,
    #[serde(default)]
    small_caps: Option<bool>,
    #[serde(default)]
    leader: Option<bool>,
    #[serde(default)]
    horizon_fade_angle: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct LineLabelPayload {
    text: String,
    polyline: Vec<[f64; 3]>,
    #[serde(default)]
    size: Option<f32>,
    #[serde(default)]
    color: Option<[f32; 4]>,
    #[serde(default)]
    halo_color: Option<[f32; 4]>,
    #[serde(default)]
    halo_width: Option<f32>,
    #[serde(default)]
    priority: Option<i32>,
    #[serde(default)]
    placement: Option<String>,
    #[serde(default)]
    repeat_distance: Option<f32>,
    #[serde(default)]
    min_zoom: Option<f32>,
    #[serde(default)]
    max_zoom: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct CurvedLabelPayload {
    text: String,
    polyline: Vec<[f64; 3]>,
    #[serde(default)]
    size: Option<f32>,
    #[serde(default)]
    color: Option<[f32; 4]>,
    #[serde(default)]
    halo_color: Option<[f32; 4]>,
    #[serde(default)]
    halo_width: Option<f32>,
    #[serde(default)]
    priority: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
struct CalloutLabelPayload {
    text: String,
    anchor: [f64; 3],
    #[serde(default)]
    offset: Option<[f32; 2]>,
    #[serde(default)]
    text_size: Option<f32>,
    #[serde(default)]
    text_color: Option<[f32; 4]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_registry_effective_state_uses_variant_preset_replacement() {
        let mut registry = ViewerSceneReviewRegistry::default();
        registry
            .install(ViewerSceneReviewStateConfig {
                base_state: ViewerSceneBaseStateConfig {
                    preset: Some(Map::from_iter([("exposure".to_string(), Value::from(1.0))])),
                    labels: vec![Map::from_iter([
                        ("kind".to_string(), Value::from("point")),
                        ("text".to_string(), Value::from("Base")),
                        (
                            "world_pos".to_string(),
                            Value::Array(vec![0.0.into(), 0.0.into(), 0.0.into()]),
                        ),
                    ])],
                    ..ViewerSceneBaseStateConfig::default()
                },
                review_layers: vec![
                    ViewerReviewLayerConfig {
                        id: "roads".to_string(),
                        labels: vec![Map::from_iter([
                            ("kind".to_string(), Value::from("point")),
                            ("text".to_string(), Value::from("Roads")),
                            (
                                "world_pos".to_string(),
                                Value::Array(vec![1.0.into(), 0.0.into(), 0.0.into()]),
                            ),
                        ])],
                        ..ViewerReviewLayerConfig::default()
                    },
                    ViewerReviewLayerConfig {
                        id: "notes".to_string(),
                        labels: vec![Map::from_iter([
                            ("kind".to_string(), Value::from("point")),
                            ("text".to_string(), Value::from("Notes")),
                            (
                                "world_pos".to_string(),
                                Value::Array(vec![2.0.into(), 0.0.into(), 0.0.into()]),
                            ),
                        ])],
                        ..ViewerReviewLayerConfig::default()
                    },
                ],
                variants: vec![ViewerSceneVariantConfig {
                    id: "review".to_string(),
                    active_layer_ids: vec!["roads".to_string()],
                    preset: Some(Map::from_iter([("exposure".to_string(), Value::from(3.0))])),
                    ..ViewerSceneVariantConfig::default()
                }],
                active_variant_id: Some("review".to_string()),
            })
            .unwrap();

        let effective = registry.effective_state().unwrap();
        assert_eq!(
            effective
                .preset
                .as_ref()
                .and_then(|preset| preset.get("exposure"))
                .and_then(Value::as_f64),
            Some(3.0)
        );
        assert_eq!(effective.labels.len(), 2);
    }

    #[test]
    fn review_registry_layer_overrides_apply_on_top_and_clear_on_variant_change() {
        let mut registry = ViewerSceneReviewRegistry::default();
        registry
            .install(ViewerSceneReviewStateConfig {
                review_layers: vec![
                    ViewerReviewLayerConfig {
                        id: "roads".to_string(),
                        ..ViewerReviewLayerConfig::default()
                    },
                    ViewerReviewLayerConfig {
                        id: "contours".to_string(),
                        ..ViewerReviewLayerConfig::default()
                    },
                ],
                variants: vec![
                    ViewerSceneVariantConfig {
                        id: "focus".to_string(),
                        active_layer_ids: vec!["roads".to_string()],
                        ..ViewerSceneVariantConfig::default()
                    },
                    ViewerSceneVariantConfig {
                        id: "analysis".to_string(),
                        active_layer_ids: vec!["contours".to_string()],
                        ..ViewerSceneVariantConfig::default()
                    },
                ],
                active_variant_id: Some("focus".to_string()),
                ..ViewerSceneReviewStateConfig::default()
            })
            .unwrap();

        registry.set_review_layer_visible("contours", true).unwrap();
        let effective = registry.effective_state().unwrap();
        assert_eq!(effective.vector_overlays.len(), 0);
        assert_eq!(registry.visible_layer_ids().len(), 2);

        registry.apply_variant("analysis").unwrap();
        assert_eq!(
            registry.state.active_variant_id.as_deref(),
            Some("analysis")
        );
        assert_eq!(
            registry.visible_layer_ids(),
            HashSet::from_iter(["contours".to_string()])
        );
    }

    #[test]
    fn review_registry_snapshot_tracks_active_variant() {
        let state = ViewerSceneReviewStateConfig {
            review_layers: vec![ViewerReviewLayerConfig {
                id: "notes".to_string(),
                name: Some("Notes".to_string()),
                ..ViewerReviewLayerConfig::default()
            }],
            variants: vec![ViewerSceneVariantConfig {
                id: "review".to_string(),
                active_layer_ids: vec!["notes".to_string()],
                ..ViewerSceneVariantConfig::default()
            }],
            active_variant_id: Some("review".to_string()),
            ..ViewerSceneReviewStateConfig::default()
        };

        let snapshot = state.snapshot();
        assert_eq!(snapshot.active_scene_variant.as_deref(), Some("review"));
        assert_eq!(snapshot.scene_variants[0].id, "review");
        assert_eq!(snapshot.review_layers[0].id, "notes");
    }
}
