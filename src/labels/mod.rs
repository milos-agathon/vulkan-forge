//! Labels module for screen-space text labels with MSDF rendering.
//!
//! Provides:
//! - `LabelManager` for managing labels lifecycle
//! - `LabelStyle` for styling configuration  
//! - Grid-based and R-tree collision detection
//! - World-to-screen projection with depth occlusion
//! - Line labels along polylines
//! - Leader lines for offset labels
//! - Scale-dependent visibility (min/max zoom)
//! - Horizon fade for labels near the horizon

mod atlas;
pub mod callout;
mod collision;
pub mod curved;
pub mod declutter;
pub mod font;
pub mod layer;
pub mod leader;
pub mod line_label;
pub mod msdf;
pub mod optimal;
pub mod positioned;
mod projection;
#[cfg(feature = "extension-module")]
pub mod py_bindings;
#[cfg(feature = "extension-module")]
pub mod py_text;
pub mod raster;
pub mod rtree;
pub mod shape;
mod types;
pub mod typography;
pub mod unicode;

pub use atlas::{GlyphKey, GlyphMetrics, MsdfAtlas};
pub use callout::{Callout, CalloutStyle, PointerDirection};
pub use collision::CollisionGrid;
pub use curved::{CurvedGlyphInstance, CurvedTextLayout, ProjectedCurvedGlyph, SampledPath};
pub use declutter::{DeclutterAlgorithm, DeclutterConfig, DeclutterResult, PlacementCandidate};
pub use layer::{
    FeatureGeometry, FeatureType, LabelFeature, LabelLayer, LabelLayerConfig, PlacementStrategy,
};
pub use leader::{create_leader_line, generate_leader_vertices};
pub use line_label::{compute_glyph_advances, compute_line_label_placement};
pub use optimal::{
    candidate_from_curved_instances, candidate_from_line_instances, candidate_from_rendered_glyphs,
    declutter_optimal, ladder_candidates, CandidateError, DepthHostAllocationReservation,
    OptimalOutcome, RationaleRecord, SolverCandidate,
};
pub use projection::LabelProjector;
pub use rtree::LabelRTree;
pub use types::{
    GlyphPlacement, LabelData, LabelFlags, LabelId, LabelStyle, LeaderLine, LineLabelData,
    LineLabelPlacement,
};
pub use typography::{KerningTable, TextCase, TypographySettings};

use crate::core::text_overlay::{TextInstance, TextOverlayRenderer};
use crate::labels::font::{FontCollection, FontRequest};
use glam::{DVec3, Mat4, Vec3};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wgpu::{Device, Queue};

fn renderer_channels_from_atlas(channels: u32) -> u32 {
    debug_assert!(matches!(channels, 1 | 3));
    channels
}

/// Stable, inspectable reason why a label was rejected during native layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelLayoutDiagnostic {
    pub label_id: LabelId,
    pub stage: &'static str,
    pub reason: String,
}

fn curved_placements_from_authority(
    text: &str,
    render_polyline: &[Vec3],
    shaped_advances: &[f32],
    font_size: f32,
    tracking: f32,
    color: [f32; 4],
    view_proj: Mat4,
    screen_width: f32,
    screen_height: f32,
) -> Result<Vec<GlyphPlacement>, String> {
    if render_polyline.len() < 2
        || render_polyline.iter().any(|point| !point.is_finite())
        || !font_size.is_finite()
        || font_size <= 0.0
        || !tracking.is_finite()
        || !screen_width.is_finite()
        || screen_width <= 0.0
        || !screen_height.is_finite()
        || screen_height <= 0.0
        || view_proj
            .to_cols_array()
            .iter()
            .any(|component| !component.is_finite())
    {
        return Err("curved layout geometry is unavailable".to_owned());
    }
    if text.chars().count() != shaped_advances.len()
        || shaped_advances
            .iter()
            .any(|advance| !advance.is_finite() || *advance < 0.0)
    {
        return Err("curved layout cannot map the shaped glyph stream one-for-one".to_owned());
    }

    let normalized_advances = shaped_advances
        .iter()
        .map(|advance| *advance / font_size)
        .collect::<Vec<_>>();
    let path = curved::SampledPath::from_polyline(render_polyline);
    let layout = curved::layout_curved_text(
        text,
        &path,
        &normalized_advances,
        font_size,
        color,
        tracking,
        true,
    );
    if !layout.success || layout.glyphs.len() != shaped_advances.len() {
        return Err("curved layout geometry is unavailable for this path".to_owned());
    }

    let projected = curved::project_curved_glyphs(&layout, view_proj, screen_width, screen_height);
    if projected.len() != layout.glyphs.len()
        || projected.iter().any(|glyph| {
            !glyph.screen_pos[0].is_finite()
                || !glyph.screen_pos[1].is_finite()
                || !glyph.rotation.is_finite()
                || !glyph.scale.is_finite()
                || glyph.scale <= 0.0
        })
    {
        return Err("curved projection geometry is unavailable".to_owned());
    }

    Ok(projected
        .into_iter()
        .map(|glyph| GlyphPlacement {
            screen_pos: glyph.screen_pos,
            rotation: glyph.rotation,
            scale: glyph.scale,
        })
        .collect())
}

fn stage_curved_rendered_instances(
    collision: &mut LabelRTree,
    rendered: &mut Vec<TextInstance>,
    label_id: LabelId,
    priority: i32,
    instances: Vec<TextInstance>,
) -> Result<Option<PlacementCandidate>, String> {
    let candidate =
        optimal::candidate_from_curved_instances(label_id.0, 0, &instances, priority)
            .ok_or_else(|| "curved atlas produced no valid rendered glyph quads".to_owned())?;
    if collision.check_collision(candidate.bounds)
        || !collision.try_insert(label_id.0, candidate.bounds)
    {
        return Ok(None);
    }
    // The candidate borrowed these exact atlas instances; move the same values
    // into the renderer stream rather than laying the glyphs out again.
    rendered.extend(instances);
    Ok(Some(candidate))
}

fn font_collection_from_metrics(
    metrics_json_path: impl AsRef<Path>,
) -> Result<Option<Arc<FontCollection>>, String> {
    let metrics_path = metrics_json_path.as_ref();
    let json = std::fs::read_to_string(metrics_path)
        .map_err(|error| format!("Failed to read atlas font metadata: {error}"))?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|error| format!("Invalid atlas font metadata JSON: {error}"))?;
    let mut sources = value
        .get("font_sources")
        .and_then(|sources| sources.as_array())
        .into_iter()
        .flatten()
        .filter_map(|source| source.as_str())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if sources.is_empty() {
        if let Some(source) = value.get("font_source").and_then(|source| source.as_str()) {
            sources.push(source.to_owned());
        }
    }
    if sources.is_empty() {
        return Ok(None);
    }

    let base = metrics_path.parent().unwrap_or_else(|| Path::new("."));
    let requests = sources
        .into_iter()
        .map(|source| {
            let declared = PathBuf::from(&source);
            let basename = declared.file_name().map(PathBuf::from);
            let path = [
                Some(declared.clone()),
                Some(base.join(&declared)),
                basename.map(|name| base.join(name)),
            ]
            .into_iter()
            .flatten()
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| format!("Atlas font source is unavailable: {source}"))?;
            let bytes = std::fs::read(&path).map_err(|error| {
                format!("Failed to read atlas font {}: {error}", path.display())
            })?;
            Ok(FontRequest::from_bytes(path.display().to_string(), bytes))
        })
        .collect::<Result<Vec<_>, String>>()?;
    FontCollection::load(&requests)
        .map(Arc::new)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Manages screen-space labels with collision detection and depth occlusion.
pub struct LabelManager {
    labels: HashMap<LabelId, LabelData>,
    line_labels: HashMap<LabelId, LineLabelData>,
    next_id: u64,
    atlas: Option<MsdfAtlas>,
    fonts: Option<Arc<FontCollection>>,
    collision_rtree: LabelRTree,
    projector: LabelProjector,
    visible_instances: Vec<TextInstance>,
    leader_lines: Vec<LeaderLine>,
    layout_diagnostics: Vec<LabelLayoutDiagnostic>,
    enabled: bool,
    current_zoom: f32,
    max_visible_labels: usize,
    typography: TypographySettings,
    declutter_algorithm: DeclutterAlgorithm,
    declutter_config: DeclutterConfig,
}

impl LabelManager {
    fn reset_layout_output(&mut self) {
        self.collision_rtree.clear();
        self.visible_instances.clear();
        self.leader_lines.clear();
        self.layout_diagnostics.clear();
        for label in self.labels.values_mut() {
            label.visible = false;
            label.screen_pos = None;
        }
        for label in self.line_labels.values_mut() {
            label.visible = false;
            label.glyph_positions.clear();
        }
    }

    /// Create a new label manager with default settings.
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            labels: HashMap::new(),
            line_labels: HashMap::new(),
            next_id: 1,
            atlas: None,
            fonts: None,
            collision_rtree: LabelRTree::new(screen_width, screen_height),
            projector: LabelProjector::new(screen_width, screen_height),
            visible_instances: Vec::new(),
            leader_lines: Vec::new(),
            layout_diagnostics: Vec::new(),
            enabled: true,
            current_zoom: 1.0,
            max_visible_labels: 500,
            typography: TypographySettings::default(),
            declutter_algorithm: DeclutterAlgorithm::default(),
            declutter_config: DeclutterConfig::default(),
        }
    }

    /// Load an MSDF atlas from font data.
    pub fn load_atlas(
        &mut self,
        device: &Device,
        queue: &Queue,
        atlas_image: &[u8],
        atlas_width: u32,
        atlas_height: u32,
        metrics_json: &str,
    ) -> Result<(), String> {
        let atlas = MsdfAtlas::load(
            device,
            queue,
            atlas_image,
            atlas_width,
            atlas_height,
            metrics_json,
        )?;
        self.atlas = Some(atlas);
        // Raw atlas bytes do not declare an immutable shaping font collection.
        // Never retain fonts from a previously loaded file-backed atlas.
        self.fonts = None;
        Ok(())
    }

    /// Load atlas from PNG file and JSON metrics file.
    pub fn load_atlas_from_files(
        &mut self,
        device: &Device,
        queue: &Queue,
        atlas_png_path: &str,
        metrics_json_path: &str,
    ) -> Result<(), String> {
        let atlas = MsdfAtlas::load_from_files(device, queue, atlas_png_path, metrics_json_path)?;
        let fonts = font_collection_from_metrics(metrics_json_path)?;
        self.atlas = Some(atlas);
        self.fonts = fonts;
        Ok(())
    }

    fn allocate_id(&mut self, requested: Option<LabelId>) -> LabelId {
        let id = requested.unwrap_or(LabelId(self.next_id));
        self.next_id = self.next_id.max(id.0.saturating_add(1));
        id
    }

    /// Add a label at a world position.
    pub fn add_label(&mut self, text: String, world_pos: Vec3, style: LabelStyle) -> LabelId {
        self.add_label_with_id(None, text, world_pos, style)
    }

    /// Add a label at a world position, preserving an externally allocated ID.
    pub fn add_label_with_id<P: Into<DVec3>>(
        &mut self,
        id: Option<LabelId>,
        text: String,
        world_pos: P,
        style: LabelStyle,
    ) -> LabelId {
        let id = self.allocate_id(id);

        let world_pos = world_pos.into();
        let label = LabelData {
            id,
            text,
            world_pos,
            render_pos: Vec3::ZERO,
            style,
            screen_pos: None,
            visible: true,
            depth: 0.0,
            horizon_angle: 0.0,
            computed_alpha: 1.0,
        };
        self.labels.insert(id, label);
        id
    }

    /// Add a line label along a polyline.
    pub fn add_line_label<P: Into<DVec3>>(
        &mut self,
        text: String,
        polyline: Vec<P>,
        style: LabelStyle,
        placement: LineLabelPlacement,
        repeat_distance: f32,
    ) -> LabelId {
        self.add_line_label_with_id(None, text, polyline, style, placement, repeat_distance)
    }

    /// Add a line label along a polyline, preserving an externally allocated ID.
    pub fn add_line_label_with_id<P: Into<DVec3>>(
        &mut self,
        id: Option<LabelId>,
        text: String,
        polyline: Vec<P>,
        style: LabelStyle,
        placement: LineLabelPlacement,
        repeat_distance: f32,
    ) -> LabelId {
        let id = self.allocate_id(id);

        let polyline: Vec<DVec3> = polyline.into_iter().map(Into::into).collect();
        let render_polyline = vec![Vec3::ZERO; polyline.len()];
        let line_label = LineLabelData {
            id,
            text,
            polyline,
            render_polyline,
            style,
            placement,
            repeat_distance,
            glyph_positions: Vec::new(),
            visible: true,
        };
        self.line_labels.insert(id, line_label);
        id
    }

    /// Remove a label by ID.
    pub fn remove_label(&mut self, id: LabelId) -> bool {
        self.labels.remove(&id).is_some() || self.line_labels.remove(&id).is_some()
    }

    /// Update label style.
    pub fn set_label_style(&mut self, id: LabelId, style: LabelStyle) -> bool {
        if let Some(label) = self.labels.get_mut(&id) {
            label.style = style;
            true
        } else {
            false
        }
    }

    /// Get a label by ID.
    pub fn get_label(&self, id: LabelId) -> Option<&LabelData> {
        self.labels.get(&id)
    }

    /// Get mutable label by ID.
    pub fn get_label_mut(&mut self, id: LabelId) -> Option<&mut LabelData> {
        self.labels.get_mut(&id)
    }

    /// Clear all labels.
    pub fn clear(&mut self) {
        self.labels.clear();
        self.line_labels.clear();
        self.reset_layout_output();
    }

    /// Set enabled state.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.reset_layout_output();
        }
    }

    /// Check if labels are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get number of labels.
    pub fn label_count(&self) -> usize {
        self.labels.len() + self.line_labels.len()
    }

    /// Absolute f64 source positions retained for prospective-frame checks.
    pub(crate) fn world_points(&self) -> Vec<DVec3> {
        self.labels
            .values()
            .map(|label| label.world_pos)
            .chain(
                self.line_labels
                    .values()
                    .flat_map(|label| label.polyline.iter().copied()),
            )
            .collect()
    }

    /// Set the current zoom level for scale-dependent visibility.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.current_zoom = zoom;
    }

    /// Get the current zoom level.
    pub fn get_zoom(&self) -> f32 {
        self.current_zoom
    }

    /// Set maximum number of visible labels.
    pub fn set_max_visible(&mut self, max: usize) {
        self.max_visible_labels = max;
    }

    /// Set global typography state for future label layout.
    pub fn set_typography(
        &mut self,
        tracking: Option<f32>,
        kerning: Option<bool>,
        line_height: Option<f32>,
        word_spacing: Option<f32>,
    ) -> TypographySettings {
        if let Some(value) = tracking {
            self.typography.tracking = value;
        }
        if let Some(value) = kerning {
            self.typography.kerning = value;
        }
        if let Some(value) = line_height {
            self.typography.line_height = value;
        }
        if let Some(value) = word_spacing {
            self.typography.word_spacing = value;
        }
        self.typography
    }

    /// Return current typography settings.
    pub fn typography(&self) -> TypographySettings {
        self.typography
    }

    /// Deterministic layout metric used by tests and validation paths.
    pub fn layout_metric_width(text: &str, font_size: f32, settings: &TypographySettings) -> f32 {
        let base_advances: Vec<f32> = text
            .chars()
            .map(|ch| if ch == ' ' { 0.3 } else { 0.5 })
            .collect();
        let mut kerning_table = KerningTable::new();
        kerning_table.load_common_latin_pairs();
        typography::compute_advances_with_typography(
            text,
            &base_advances,
            font_size,
            settings,
            Some(&kerning_table),
        )
        .iter()
        .sum()
    }

    /// Set label declutter policy state.
    pub fn set_declutter_algorithm(
        &mut self,
        algorithm: DeclutterAlgorithm,
        seed: Option<u64>,
        max_iterations: Option<usize>,
    ) -> (DeclutterAlgorithm, DeclutterConfig) {
        self.declutter_algorithm = algorithm;
        if let Some(value) = seed {
            self.declutter_config.seed = value;
        }
        if let Some(value) = max_iterations {
            self.declutter_config.max_iterations = value;
        }
        (self.declutter_algorithm, self.declutter_config.clone())
    }

    /// Resize for new screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.collision_rtree.resize(width, height);
        self.projector = LabelProjector::new(width, height);
    }

    /// Get leader lines for rendering.
    pub fn leader_lines(&self) -> &[LeaderLine] {
        &self.leader_lines
    }

    /// Structured failures from the most recent layout transaction.
    pub fn layout_diagnostics(&self) -> &[LabelLayoutDiagnostic] {
        &self.layout_diagnostics
    }

    /// Pick a label at the given screen coordinates.
    pub fn pick_at(&self, x: f32, y: f32) -> Option<LabelId> {
        // Query a small box around the cursor (e.g. 4x4 pixels)
        let bounds = [x - 2.0, y - 2.0, x + 2.0, y + 2.0];
        let hits = self.collision_rtree.query_intersecting(bounds);

        // Return the first hit
        // In a real implementation we might want to sort by depth or priority
        hits.first().map(|h| LabelId(h.id))
    }

    /// Update label positions and visibility based on current view.
    /// Returns the number of visible labels.
    pub fn update(&mut self, view_proj: Mat4) -> usize {
        self.update_with_camera(view_proj, None, None)
    }

    /// Update with camera position for horizon fade calculation.
    pub fn update_with_camera(
        &mut self,
        view_proj: Mat4,
        camera_pos: Option<Vec3>,
        selected_ids: Option<&std::collections::HashSet<u64>>,
    ) -> usize {
        self.update_with_camera_anchored(
            view_proj,
            camera_pos,
            selected_ids,
            &crate::camera::Anchor::new(),
        )
    }

    /// Update using the viewer's frozen frame anchor.
    pub fn update_with_camera_anchored(
        &mut self,
        view_proj: Mat4,
        camera_pos: Option<Vec3>,
        selected_ids: Option<&std::collections::HashSet<u64>>,
        anchor: &crate::camera::Anchor,
    ) -> usize {
        struct StagedLabelState {
            screen_pos: [f32; 2],
            depth: f32,
            horizon_angle: f32,
            computed_alpha: f32,
            render_pos: Vec3,
        }

        struct StagedLineState {
            glyph_positions: Vec<GlyphPlacement>,
            render_polyline: Vec<Vec3>,
        }

        if !self.enabled {
            self.reset_layout_output();
            return 0;
        }

        let atlas = self.atlas.as_ref();

        let (screen_w, screen_h) = self.projector.screen_size();
        let mut visible_count = 0;
        let fonts = self.fonts.clone();
        let mut staged_collision =
            LabelRTree::new(screen_w.round() as u32, screen_h.round() as u32);
        let mut staged_instances = Vec::new();
        let mut staged_leader_lines = Vec::new();
        let mut staged_diagnostics = Vec::new();
        let mut staged_labels: HashMap<LabelId, StagedLabelState> = HashMap::new();
        let mut staged_line_labels: HashMap<LabelId, StagedLineState> = HashMap::new();

        // Collect labels and sort by priority (higher priority first)
        let mut sorted_labels: Vec<_> = self.labels.values().collect();
        sorted_labels.sort_by_key(|label| std::cmp::Reverse(label.style.priority));

        for label in sorted_labels {
            let render_pos = anchor.to_render_vec3(label.world_pos);
            // Skip if we've reached max visible
            if visible_count >= self.max_visible_labels {
                continue;
            }

            // Scale filtering: check zoom range
            if self.current_zoom < label.style.min_zoom || self.current_zoom > label.style.max_zoom
            {
                continue;
            }

            // Project world position to screen
            let projected = self.projector.project(render_pos, view_proj);

            if let Some((mut screen_pos, depth)) = projected {
                let staged_depth = depth;
                let staged_horizon_angle;
                let mut staged_computed_alpha;

                // Compute horizon angle for fade without mutating label state
                let horizon_alpha = if let Some(cam_pos) = camera_pos {
                    let to_label = render_pos - cam_pos;
                    let horizontal_dist =
                        (to_label.x * to_label.x + to_label.z * to_label.z).sqrt();
                    let angle_deg = (to_label.y / horizontal_dist.max(0.001))
                        .atan()
                        .to_degrees();
                    staged_horizon_angle = angle_deg;

                    // Fade based on horizon angle
                    let fade_start = label.style.horizon_fade_angle;
                    if angle_deg.abs() < fade_start {
                        (angle_deg.abs() / fade_start).clamp(0.0, 1.0)
                    } else {
                        1.0
                    }
                } else {
                    staged_horizon_angle = 90.0;
                    1.0
                };

                staged_computed_alpha = horizon_alpha * label.style.color[3];

                // Apply offset
                let anchor_screen = screen_pos;
                screen_pos[0] += label.style.offset[0];
                screen_pos[1] += label.style.offset[1];

                // Shape once and keep the exact fallback face / GSUB glyph identities
                // for both collision bounds and GPU instance construction.
                let shaped = if let Some(fonts) = &fonts {
                    match shape::shape(
                        &label.text,
                        Arc::clone(fonts),
                        label.style.size,
                        None,
                        None,
                        &[],
                    ) {
                        Ok(shaped) => Some(shaped),
                        Err(error) => {
                            staged_diagnostics.push(LabelLayoutDiagnostic {
                                label_id: label.id,
                                stage: "shape",
                                reason: error.to_string(),
                            });
                            continue;
                        }
                    }
                } else {
                    None
                };
                let line_range = 0..label.text.chars().count();
                let line_ranges = std::slice::from_ref(&line_range);
                let (width, height) = if let (Some(atlas), Some(shaped)) = (atlas, &shaped) {
                    match atlas.measure_shaped(shaped, line_ranges) {
                        Ok(measurement) => measurement,
                        Err(error) => {
                            staged_diagnostics.push(LabelLayoutDiagnostic {
                                label_id: label.id,
                                stage: "measure",
                                reason: error,
                            });
                            continue;
                        }
                    }
                } else if let Some(atlas) = atlas {
                    atlas.measure_text(&label.text, label.style.size)
                } else {
                    let glyphs = label.text.chars().filter(|ch| !ch.is_whitespace()).count();
                    (
                        (glyphs.max(1) as f32) * label.style.size * 0.6,
                        label.style.size,
                    )
                };
                let half_w = width * 0.5;
                let half_h = height * 0.5;

                let bounds = [
                    screen_pos[0] - half_w,
                    screen_pos[1] - half_h,
                    screen_pos[0] + half_w,
                    screen_pos[1] + half_h,
                ];

                // Preflight the complete GPU instance stream before mutating
                // collision state, visibility counts, or leader-line output.
                let mut color = label.style.color;
                if let Some(selected) = selected_ids {
                    if selected.contains(&label.id.0) {
                        color = [1.0, 0.8, 0.0, 1.0];
                        staged_computed_alpha = 1.0;
                    }
                }
                color[3] = staged_computed_alpha;
                let instances = if let (Some(atlas), Some(shaped)) = (atlas, &shaped) {
                    match atlas.layout_shaped(
                        shaped,
                        line_ranges,
                        screen_pos,
                        color,
                        label.style.halo_color,
                        label.style.halo_width,
                    ) {
                        Ok(instances) => instances,
                        Err(error) => {
                            staged_diagnostics.push(LabelLayoutDiagnostic {
                                label_id: label.id,
                                stage: "layout",
                                reason: error,
                            });
                            continue;
                        }
                    }
                } else if let Some(atlas) = atlas {
                    atlas.layout_text(
                        &label.text,
                        screen_pos,
                        label.style.size,
                        color,
                        label.style.halo_color,
                        label.style.halo_width,
                    )
                } else {
                    Vec::new()
                };

                if staged_collision.check_collision(bounds) {
                    continue;
                }
                if !staged_collision.try_insert(label.id.0, bounds) {
                    continue;
                }

                staged_labels.insert(
                    label.id,
                    StagedLabelState {
                        screen_pos,
                        depth: staged_depth,
                        horizon_angle: staged_horizon_angle,
                        computed_alpha: staged_computed_alpha,
                        render_pos,
                    },
                );
                visible_count += 1;
                if label.style.flags.leader
                    && (label.style.offset[0].abs() > 1.0 || label.style.offset[1].abs() > 1.0)
                {
                    staged_leader_lines.push(create_leader_line(
                        anchor_screen,
                        screen_pos,
                        label.style.halo_color,
                        1.5,
                    ));
                }
                staged_instances.extend(instances);
            }
        }

        // Process line labels
        for line_label in self.line_labels.values() {
            if visible_count >= self.max_visible_labels {
                continue;
            }

            // Scale filtering
            if self.current_zoom < line_label.style.min_zoom
                || self.current_zoom > line_label.style.max_zoom
            {
                continue;
            }

            let render_polyline = line_label
                .polyline
                .iter()
                .copied()
                .map(|world| anchor.to_render_vec3(world))
                .collect::<Vec<_>>();
            let is_curved = line_label.placement == LineLabelPlacement::Along;
            let geometry_kind = if is_curved { "curved" } else { "line" };

            let Some(atlas) = atlas else {
                staged_diagnostics.push(LabelLayoutDiagnostic {
                    label_id: line_label.id,
                    stage: "authority",
                    reason: format!("{geometry_kind} glyph bounds require a loaded atlas"),
                });
                continue;
            };
            let Some(fonts) = &fonts else {
                staged_diagnostics.push(LabelLayoutDiagnostic {
                    label_id: line_label.id,
                    stage: "authority",
                    reason: format!("{geometry_kind} glyph bounds require shaping fonts"),
                });
                continue;
            };
            let shaped = match shape::shape(
                &line_label.text,
                Arc::clone(fonts),
                line_label.style.size,
                None,
                None,
                &[],
            ) {
                Ok(shaped) => shaped,
                Err(error) => {
                    staged_diagnostics.push(LabelLayoutDiagnostic {
                        label_id: line_label.id,
                        stage: "shape",
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            let line_range = 0..line_label.text.chars().count();
            let line_ranges = std::slice::from_ref(&line_range);
            let shaped_stream = match positioned::positioned_glyphs(&shaped, line_ranges) {
                Ok(glyphs) => glyphs,
                Err(error) => {
                    staged_diagnostics.push(LabelLayoutDiagnostic {
                        label_id: line_label.id,
                        stage: "position",
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            let advances = shaped_stream
                .iter()
                .map(|glyph| glyph.advance[0])
                .collect::<Vec<_>>();

            // `Along` is the production curved-label representation used by
            // AddCurvedLabel. Its existing layout/projection authority places
            // the shaped stream; `Center` retains the straight-line authority.
            let placements = if is_curved {
                match curved_placements_from_authority(
                    &line_label.text,
                    &render_polyline,
                    &advances,
                    line_label.style.size,
                    self.typography.tracking,
                    line_label.style.color,
                    view_proj,
                    screen_w,
                    screen_h,
                ) {
                    Ok(placements) => placements,
                    Err(reason) => {
                        staged_diagnostics.push(LabelLayoutDiagnostic {
                            label_id: line_label.id,
                            stage: "authority",
                            reason,
                        });
                        continue;
                    }
                }
            } else {
                compute_line_label_placement(
                    &render_polyline,
                    &line_label.text,
                    &advances,
                    view_proj,
                    screen_w,
                    screen_h,
                    LineLabelPlacement::Center,
                    line_label.style.size,
                )
            };

            if placements.is_empty() {
                staged_diagnostics.push(LabelLayoutDiagnostic {
                    label_id: line_label.id,
                    stage: "placement",
                    reason: "line has no valid glyph placements".to_owned(),
                });
                continue;
            }
            let mut color = line_label.style.color;
            color[3] = color[3].clamp(0.0, 1.0);
            let instances = match atlas.layout_shaped_on_placements(
                &shaped,
                line_ranges,
                &placements,
                color,
                line_label.style.halo_color,
                line_label.style.halo_width,
            ) {
                Ok(instances) => instances,
                Err(error) => {
                    staged_diagnostics.push(LabelLayoutDiagnostic {
                        label_id: line_label.id,
                        stage: if is_curved { "authority" } else { "layout" },
                        reason: if is_curved {
                            format!("curved atlas glyph geometry is unavailable: {error}")
                        } else {
                            error
                        },
                    });
                    continue;
                }
            };
            if is_curved {
                match stage_curved_rendered_instances(
                    &mut staged_collision,
                    &mut staged_instances,
                    line_label.id,
                    line_label.style.priority,
                    instances,
                ) {
                    Ok(Some(_candidate)) => {}
                    Ok(None) => continue,
                    Err(reason) => {
                        staged_diagnostics.push(LabelLayoutDiagnostic {
                            label_id: line_label.id,
                            stage: "authority",
                            reason,
                        });
                        continue;
                    }
                }
            } else {
                let placement_candidate = match optimal::candidate_from_line_instances(
                    line_label.id.0,
                    0,
                    &instances,
                    line_label.style.priority,
                ) {
                    Some(candidate) => candidate,
                    None => {
                        staged_diagnostics.push(LabelLayoutDiagnostic {
                            label_id: line_label.id,
                            stage: "bounds",
                            reason: "atlas/shaping produced no valid rendered glyph quads"
                                .to_owned(),
                        });
                        continue;
                    }
                };
                if staged_collision.check_collision(placement_candidate.bounds)
                    || !staged_collision.try_insert(line_label.id.0, placement_candidate.bounds)
                {
                    continue;
                }
                staged_instances.extend(instances);
            }
            staged_line_labels.insert(
                line_label.id,
                StagedLineState {
                    glyph_positions: placements,
                    render_polyline,
                },
            );
            visible_count += 1;
        }

        for label in self.labels.values_mut() {
            label.visible = false;
            label.screen_pos = None;
        }
        for line_label in self.line_labels.values_mut() {
            line_label.visible = false;
            line_label.glyph_positions.clear();
        }
        for (id, state) in staged_labels {
            if let Some(label) = self.labels.get_mut(&id) {
                label.depth = state.depth;
                label.horizon_angle = state.horizon_angle;
                label.computed_alpha = state.computed_alpha;
                label.render_pos = state.render_pos;
                label.screen_pos = Some(state.screen_pos);
                label.visible = true;
            }
        }
        for (id, state) in staged_line_labels {
            if let Some(line_label) = self.line_labels.get_mut(&id) {
                line_label.glyph_positions = state.glyph_positions;
                line_label.render_polyline = state.render_polyline;
                line_label.visible = true;
            }
        }
        self.collision_rtree = staged_collision;
        self.visible_instances = staged_instances;
        self.leader_lines = staged_leader_lines;
        self.layout_diagnostics = staged_diagnostics;

        self.visible_instances.len()
    }

    /// Upload instances to the text overlay renderer.
    pub fn upload_to_renderer(
        &self,
        device: &Device,
        queue: &Queue,
        renderer: &mut TextOverlayRenderer,
    ) {
        if let Some(atlas) = &self.atlas {
            // Recreate bind group with the atlas view. Best-effort HUD label path:
            // the internal allocations were infallible before the tracked-wrapper
            // migration, so a rare allocation failure is discarded rather than
            // aborting the frame (this fn returns a count, not a Result).
            let _ = renderer.recreate_bind_group(device, Some(&atlas.view));
        }

        renderer.set_channels(
            self.atlas
                .as_ref()
                .map_or(3, |atlas| renderer_channels_from_atlas(atlas.channels)),
        );
        renderer.set_smoothing(2.0);

        let _ = renderer.upload_instances(device, queue, &self.visible_instances);
    }

    /// Get reference to the atlas view if loaded.
    pub fn atlas_view(&self) -> Option<&wgpu::TextureView> {
        self.atlas.as_ref().map(|a| a.view.as_ref())
    }

    /// Get visible instance count.
    pub fn visible_count(&self) -> usize {
        self.visible_instances.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_manager_typography_and_declutter_state_mutate() {
        let mut manager = LabelManager::new(800, 600);

        let typography = manager.set_typography(Some(0.25), Some(true), Some(1.3), Some(2.0));
        assert_eq!(typography.tracking, 0.25);
        assert!(typography.kerning);
        assert_eq!(typography.line_height, 1.3);
        assert_eq!(typography.word_spacing, 2.0);

        let default_width =
            LabelManager::layout_metric_width("AV label", 16.0, &TypographySettings::default());
        let typography_width = LabelManager::layout_metric_width("AV label", 16.0, &typography);
        assert!(typography_width > default_width);

        let (algorithm, config) =
            manager.set_declutter_algorithm(DeclutterAlgorithm::Annealing, Some(123), Some(50));
        assert_eq!(algorithm, DeclutterAlgorithm::Annealing);
        assert_eq!(config.seed, 123);
        assert_eq!(config.max_iterations, 50);
    }

    #[test]
    fn updating_without_an_atlas_keeps_labels_pickable_with_fallback_bounds() {
        let mut manager = LabelManager::new(800, 600);
        let id = manager.add_label("Map".to_owned(), Vec3::ZERO, LabelStyle::default());
        assert!(manager.get_label(id).unwrap().visible);

        assert_eq!(manager.update(Mat4::IDENTITY), 0);
        let label = manager.get_label(id).unwrap();
        assert!(label.visible);
        assert_eq!(label.screen_pos, Some([400.0, 300.0]));
        assert_eq!(manager.visible_count(), 0);
        assert_eq!(manager.pick_at(400.0, 300.0), Some(id));

        manager.set_enabled(false);
        assert!(!manager.get_label(id).unwrap().visible);
        assert!(manager.get_label(id).unwrap().screen_pos.is_none());
        assert!(manager.pick_at(400.0, 300.0).is_none());
    }

    #[test]
    fn curved_and_line_labels_report_distinct_missing_authority_diagnostics() {
        let mut manager = LabelManager::new(800, 600);
        let curved_id = manager.add_line_label(
            "Road".to_owned(),
            vec![Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)],
            LabelStyle::default(),
            LineLabelPlacement::Along,
            0.0,
        );
        let line_id = manager.add_line_label(
            "Street".to_owned(),
            vec![Vec3::new(-0.5, 0.2, 0.0), Vec3::new(0.5, 0.2, 0.0)],
            LabelStyle::default(),
            LineLabelPlacement::Center,
            0.0,
        );
        assert_eq!(manager.update(Mat4::IDENTITY), 0);
        assert!(manager.layout_diagnostics().iter().any(|diagnostic| {
            diagnostic.label_id == curved_id
                && diagnostic.stage == "authority"
                && diagnostic.reason == "curved glyph bounds require a loaded atlas"
        }));
        assert!(manager.layout_diagnostics().iter().any(|diagnostic| {
            diagnostic.label_id == line_id
                && diagnostic.stage == "authority"
                && diagnostic.reason == "line glyph bounds require a loaded atlas"
        }));
    }

    #[test]
    fn production_curved_path_reuses_exact_rendered_quads_for_collision_and_render() {
        let view_proj = Mat4::from_scale(Vec3::new(0.005, 0.01, 1.0));
        let placements = curved_placements_from_authority(
            "AB",
            &[
                Vec3::new(-100.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(100.0, 0.0, 0.0),
            ],
            &[20.0, 30.0],
            20.0,
            0.0,
            [0.2, 0.4, 0.6, 1.0],
            view_proj,
            800.0,
            600.0,
        )
        .expect("curved layout and nonidentity projection must produce glyph placements");
        assert_eq!(placements.len(), 2);

        // These are the glyph-specific quads that the production branch gets
        // from MsdfAtlas::layout_shaped_on_placements: deliberately distinct
        // dimensions, bearings, UVs, halos, and rotations.
        let mut first = TextInstance::new(
            [
                placements[0].screen_pos[0] - 3.0,
                placements[0].screen_pos[1] - 7.0,
            ],
            [
                placements[0].screen_pos[0] + 9.0,
                placements[0].screen_pos[1] + 5.0,
            ],
            [0.0, 0.0],
            [0.25, 0.5],
            [0.2, 0.4, 0.6, 1.0],
        )
        .with_halo([1.0, 0.5, 0.25, 0.75], 1.25);
        first.rotation = placements[0].rotation;
        let mut second = TextInstance::new(
            [
                placements[1].screen_pos[0] - 8.0,
                placements[1].screen_pos[1] - 4.0,
            ],
            [
                placements[1].screen_pos[0] + 5.0,
                placements[1].screen_pos[1] + 10.0,
            ],
            [0.25, 0.0],
            [0.75, 0.5],
            [0.2, 0.4, 0.6, 1.0],
        )
        .with_halo([1.0, 0.5, 0.25, 0.75], 1.25);
        second.rotation = placements[1].rotation;
        let atlas_instances = vec![first, second];
        let exact_atlas_bytes = atlas_instances
            .iter()
            .map(bytemuck::bytes_of)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();

        let mut collision = LabelRTree::new(800, 600);
        let mut rendered = Vec::new();
        let candidate = stage_curved_rendered_instances(
            &mut collision,
            &mut rendered,
            LabelId(77),
            9,
            atlas_instances,
        )
        .expect("atlas quads are valid curved geometry")
        .expect("curved geometry must fit the empty collision tree");

        assert_eq!(rendered.len(), exact_atlas_bytes.len());
        for (instance, expected) in rendered.iter().zip(&exact_atlas_bytes) {
            assert_eq!(bytemuck::bytes_of(instance), expected);
        }
        let hits = collision.query_intersecting(candidate.bounds);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 77);
        assert_eq!(hits[0].bounds, candidate.bounds);
        let rendered_candidate = optimal::candidate_from_curved_instances(77, 0, &rendered, 9)
            .expect("the unchanged render stream remains valid collision geometry");
        assert_eq!(rendered_candidate.bounds, candidate.bounds);
    }

    #[test]
    fn label_manager_layout_writes_to_staged_collision_before_commit() {
        let source = include_str!("mod.rs");
        let persistent_insert = ["self.collision_rtree", ".try_insert"].concat();

        assert!(
            !source.contains(&persistent_insert),
            "layout must not mutate the persistent collision tree before commit"
        );
        assert!(source.contains("staged_collision.try_insert"));
        assert!(source.contains("self.collision_rtree = staged_collision"));
        assert!(source.contains("self.visible_instances = staged_instances"));
        assert!(source.contains("self.leader_lines = staged_leader_lines"));
        assert!(source.contains("self.layout_diagnostics = staged_diagnostics"));
    }
}
