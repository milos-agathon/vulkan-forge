//! MSDF font atlas loading and text layout.

use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::core::text_overlay::TextInstance;
use crate::labels::positioned::{positioned_glyphs, PositionedGlyph};
use crate::labels::shape::ShapedText;
use crate::labels::types::GlyphPlacement;
use serde::Deserialize;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use wgpu::{Device, Queue, TextureView};

/// Metrics for a single glyph in the atlas.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    /// Unicode codepoint.
    pub codepoint: u32,
    /// Source face in the immutable shaped font collection.
    pub font_index: usize,
    /// Font-specific glyph identifier after GSUB/fallback selection.
    pub glyph_id: u16,
    /// UV coordinates in atlas [u0, v0, u1, v1].
    pub uv: [f32; 4],
    /// Glyph width in atlas pixels.
    pub width: f32,
    /// Glyph height in atlas pixels.
    pub height: f32,
    /// Horizontal offset from cursor to glyph origin.
    pub offset_x: f32,
    /// Vertical offset from baseline to glyph top.
    pub offset_y: f32,
    /// Horizontal advance after this glyph.
    pub advance: f32,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct GlyphKey {
    pub font_index: usize,
    pub glyph_id: u16,
}

fn path_glyph_center(
    glyph: &PositionedGlyph,
    metric: &GlyphMetrics,
    placement: &GlyphPlacement,
    pen: f32,
    scale: f32,
    shaped_size: f32,
) -> [f32; 2] {
    let half_width = metric.width * scale * 0.5;
    let half_height = metric.height * scale * 0.5;
    let baseline_y = glyph.line_index as f32 * shaped_size * 1.2;
    let local_x =
        glyph.origin[0] - pen + metric.offset_x * scale + half_width - glyph.advance[0] * 0.5;
    let local_y = glyph.origin[1] - baseline_y + metric.offset_y * scale + half_height;
    let (sin, cos) = placement.rotation.sin_cos();
    [
        placement.screen_pos[0] + local_x * cos - local_y * sin,
        placement.screen_pos[1] + local_x * sin + local_y * cos,
    ]
}

/// MSDF font atlas with glyph metrics.
pub struct MsdfAtlas {
    pub texture: Arc<TrackedTexture>,
    pub view: Arc<TextureView>,
    pub width: u32,
    pub height: u32,
    /// Canonical glyph metrics indexed by shaped font/glyph identity.
    glyphs: HashMap<GlyphKey, GlyphMetrics>,
    /// Compatibility aliases from source Unicode codepoints to shaped identities.
    unicode_map: HashMap<u32, GlyphKey>,
    /// Font size used when generating the atlas.
    pub atlas_font_size: f32,
    /// Line height in atlas pixels.
    pub line_height: f32,
    /// Baseline offset from top of line.
    pub baseline: f32,
    /// Distance-field channels declared by atlas metadata (1=SDF, 3=MSDF).
    pub channels: u32,
}

#[derive(Debug, Deserialize)]
struct AtlasMetricsJson {
    font_size: Option<f32>,
    line_height: Option<f32>,
    baseline: Option<f32>,
    channels: Option<u32>,
    #[serde(default)]
    glyphs: HashMap<String, GlyphMetricsJson>,
    #[serde(default)]
    glyphs_by_id: HashMap<String, GlyphMetricsJson>,
    #[serde(default)]
    unicode_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct GlyphMetricsJson {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    #[serde(default)]
    ox: f32,
    #[serde(default)]
    oy: f32,
    adv: Option<f32>,
    font_index: Option<usize>,
    glyph_id: Option<u16>,
}

impl MsdfAtlas {
    /// Load an MSDF atlas from raw image data and JSON metrics.
    pub fn load(
        device: &Device,
        queue: &Queue,
        atlas_image: &[u8],
        atlas_width: u32,
        atlas_height: u32,
        metrics_json: &str,
    ) -> Result<Self, String> {
        let (glyphs, unicode_map, atlas_font_size, line_height, baseline, channels) =
            Self::parse_metrics(metrics_json, atlas_width, atlas_height)?;
        let pixel_count = atlas_width as usize * atlas_height as usize;
        let upload_data = if channels == 3 && atlas_image.len() == pixel_count * 3 {
            atlas_image
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                .collect::<Vec<_>>()
        } else if atlas_image.len() == pixel_count * 4 {
            atlas_image.to_vec()
        } else {
            return Err(format!(
                "Atlas byte count {} does not match {}x{} with {} channels",
                atlas_image.len(),
                atlas_width,
                atlas_height,
                channels
            ));
        };
        // Allocate the atlas texture through the tracked wrapper.
        let texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("msdf_atlas"),
                size: wgpu::Extent3d {
                    width: atlas_width,
                    height: atlas_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
        .map_err(|e| e.to_string())?;

        // Upload image data
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * atlas_width),
                rows_per_image: Some(atlas_height),
            },
            wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            texture: Arc::new(texture),
            view: Arc::new(view),
            width: atlas_width,
            height: atlas_height,
            glyphs,
            unicode_map,
            atlas_font_size,
            line_height,
            baseline,
            channels,
        })
    }

    /// Load atlas from PNG file and JSON metrics file.
    pub fn load_from_files(
        device: &Device,
        queue: &Queue,
        atlas_png_path: &str,
        metrics_json_path: &str,
    ) -> Result<Self, String> {
        // Load PNG using the image crate
        let img =
            image::open(atlas_png_path).map_err(|e| format!("Failed to load atlas PNG: {}", e))?;

        let width = img.width();
        let height = img.height();
        // Load JSON metrics
        let metrics_json = std::fs::read_to_string(metrics_json_path)
            .map_err(|e| format!("Failed to read metrics JSON: {}", e))?;
        let channels = serde_json::from_str::<AtlasMetricsJson>(&metrics_json)
            .map_err(|error| format!("Invalid atlas metrics JSON: {error}"))?
            .channels
            .unwrap_or(1);
        let image_data = if channels == 3 {
            img.to_rgb8().into_raw()
        } else {
            img.to_rgba8().into_raw()
        };

        Self::load(device, queue, &image_data, width, height, &metrics_json)
    }

    /// Parse metrics from JSON.
    /// Supports a simplified format:
    /// {
    ///   "font_size": 32,
    ///   "line_height": 40,
    ///   "baseline": 32,
    ///   "glyphs": {
    ///     "65": { "x": 0, "y": 0, "w": 20, "h": 30, "ox": 0, "oy": 0, "adv": 22 },
    ///     ...
    ///   }
    /// }
    fn parse_metrics(
        json: &str,
        atlas_width: u32,
        atlas_height: u32,
    ) -> Result<
        (
            HashMap<GlyphKey, GlyphMetrics>,
            HashMap<u32, GlyphKey>,
            f32,
            f32,
            f32,
            u32,
        ),
        String,
    > {
        let parsed: AtlasMetricsJson =
            serde_json::from_str(json).map_err(|e| format!("Invalid atlas metrics JSON: {e}"))?;
        if parsed.glyphs.is_empty() && parsed.glyphs_by_id.is_empty() {
            return Err("Atlas metrics must contain at least one glyph".to_string());
        }

        let font_size = parsed.font_size.unwrap_or(32.0);
        let line_height = parsed.line_height.unwrap_or(font_size * 1.25);
        let baseline = parsed.baseline.unwrap_or(font_size);
        let channels = parsed.channels.unwrap_or(1);
        if channels != 1 && channels != 3 {
            return Err(format!("Atlas channels must be 1 or 3, got {channels}"));
        }
        let parse_key = |value: &str| -> Result<GlyphKey, String> {
            let (font_index, glyph_id) = value
                .split_once(':')
                .ok_or_else(|| format!("Atlas glyph identity must be font:glyph, got {value:?}"))?;
            Ok(GlyphKey {
                font_index: font_index
                    .parse()
                    .map_err(|_| format!("Invalid atlas font index: {font_index:?}"))?,
                glyph_id: glyph_id
                    .parse()
                    .map_err(|_| format!("Invalid atlas glyph id: {glyph_id:?}"))?,
            })
        };
        let metric = |codepoint: u32,
                      key: GlyphKey,
                      glyph: &GlyphMetricsJson|
         -> Result<GlyphMetrics, String> {
            if glyph.w <= 0.0 || glyph.h <= 0.0 {
                return Err(format!(
                    "Atlas glyph {}:{} has non-positive dimensions",
                    key.font_index, key.glyph_id
                ));
            }
            Ok(GlyphMetrics {
                codepoint,
                font_index: key.font_index,
                glyph_id: key.glyph_id,
                uv: [
                    glyph.x / atlas_width as f32,
                    glyph.y / atlas_height as f32,
                    (glyph.x + glyph.w) / atlas_width as f32,
                    (glyph.y + glyph.h) / atlas_height as f32,
                ],
                width: glyph.w,
                height: glyph.h,
                offset_x: glyph.ox,
                offset_y: glyph.oy,
                advance: glyph.adv.unwrap_or(glyph.w),
            })
        };
        let mut glyphs = HashMap::with_capacity(parsed.glyphs.len() + parsed.glyphs_by_id.len());
        let mut unicode_aliases = HashMap::with_capacity(parsed.glyphs.len());

        for (identity, glyph) in &parsed.glyphs_by_id {
            let key = parse_key(identity)?;
            if glyph
                .font_index
                .is_some_and(|value| value != key.font_index)
                || glyph.glyph_id.is_some_and(|value| value != key.glyph_id)
            {
                return Err(format!(
                    "Atlas glyph identity {identity:?} disagrees with its metric fields"
                ));
            }
            glyphs.insert(key, metric(0, key, glyph)?);
        }

        for (codepoint_str, glyph) in &parsed.glyphs {
            let codepoint = codepoint_str.parse::<u32>().map_err(|_| {
                format!("Atlas glyph key is not a Unicode codepoint: {codepoint_str}")
            })?;
            let key = if let Some(identity) = parsed.unicode_map.get(codepoint_str) {
                parse_key(identity)?
            } else if let (Some(font_index), Some(glyph_id)) = (glyph.font_index, glyph.glyph_id) {
                GlyphKey {
                    font_index,
                    glyph_id,
                }
            } else {
                GlyphKey {
                    font_index: 0,
                    glyph_id: u16::try_from(codepoint).map_err(|_| {
                        format!("Legacy atlas glyph U+{codepoint:04X} needs an explicit glyph_id")
                    })?,
                }
            };
            unicode_aliases.insert(codepoint, key);
            glyphs.entry(key).or_insert(metric(codepoint, key, glyph)?);
        }
        for (codepoint, identity) in &parsed.unicode_map {
            let codepoint = codepoint
                .parse::<u32>()
                .map_err(|_| format!("Atlas unicode_map key is not a codepoint: {codepoint:?}"))?;
            let key = parse_key(identity)?;
            if !glyphs.contains_key(&key) {
                return Err(format!(
                    "Atlas unicode_map U+{codepoint:04X} references missing identity {identity:?}"
                ));
            }
            unicode_aliases.insert(codepoint, key);
            if let Some(glyph) = glyphs.get_mut(&key) {
                if glyph.codepoint == 0 {
                    glyph.codepoint = codepoint;
                }
            }
        }

        Ok((
            glyphs,
            unicode_aliases,
            font_size,
            line_height,
            baseline,
            channels,
        ))
    }

    /// Get glyph metrics for a character.
    pub fn get_glyph(&self, c: char) -> Option<&GlyphMetrics> {
        self.unicode_map
            .get(&(c as u32))
            .and_then(|key| self.glyphs.get(key))
    }

    pub fn get_glyph_id(&self, font_index: usize, glyph_id: u16) -> Option<&GlyphMetrics> {
        self.glyphs.get(&GlyphKey {
            font_index,
            glyph_id,
        })
    }

    fn shaped_rects(
        &self,
        shaped: &ShapedText,
        line_ranges: &[Range<usize>],
    ) -> Result<Vec<([f32; 4], GlyphMetrics)>, String> {
        let scale = shaped.size / self.atlas_font_size;
        positioned_glyphs(shaped, line_ranges)
            .map_err(|error| error.to_string())?
            .into_iter()
            .filter(|glyph| glyph.path.is_some())
            .map(|glyph| {
                let metric = self
                    .get_glyph_id(glyph.font_index, glyph.glyph_id)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "Atlas is missing shaped glyph {}:{}",
                            glyph.font_index, glyph.glyph_id
                        )
                    })?;
                let x0 = glyph.origin[0] + metric.offset_x * scale;
                let y0 = glyph.origin[1] + metric.offset_y * scale;
                Ok((
                    [
                        x0,
                        y0,
                        x0 + metric.width * scale,
                        y0 + metric.height * scale,
                    ],
                    metric,
                ))
            })
            .collect()
    }

    /// Measure the positioned shaped-glyph stream rather than iterating codepoints.
    pub fn measure_shaped(
        &self,
        shaped: &ShapedText,
        line_ranges: &[Range<usize>],
    ) -> Result<(f32, f32), String> {
        let rects = self.shaped_rects(shaped, line_ranges)?;
        if rects.is_empty() {
            return Ok((0.0, shaped.size));
        }
        let bounds = rects.iter().fold(
            [
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            |mut bounds, (rect, _)| {
                bounds[0] = bounds[0].min(rect[0]);
                bounds[1] = bounds[1].min(rect[1]);
                bounds[2] = bounds[2].max(rect[2]);
                bounds[3] = bounds[3].max(rect[3]);
                bounds
            },
        );
        Ok((
            bounds[2] - bounds[0],
            (bounds[3] - bounds[1]).max(shaped.size),
        ))
    }

    /// Build GPU quads from visual `PositionedGlyph` records keyed by font/glyph id.
    pub fn layout_shaped(
        &self,
        shaped: &ShapedText,
        line_ranges: &[Range<usize>],
        center_pos: [f32; 2],
        color: [f32; 4],
        halo_color: [f32; 4],
        halo_width: f32,
    ) -> Result<Vec<TextInstance>, String> {
        let rects = self.shaped_rects(shaped, line_ranges)?;
        if rects.is_empty() {
            return Ok(Vec::new());
        }
        let bounds = rects.iter().fold(
            [
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            |mut bounds, (rect, _)| {
                bounds[0] = bounds[0].min(rect[0]);
                bounds[1] = bounds[1].min(rect[1]);
                bounds[2] = bounds[2].max(rect[2]);
                bounds[3] = bounds[3].max(rect[3]);
                bounds
            },
        );
        let shift = [
            center_pos[0] - (bounds[0] + bounds[2]) * 0.5,
            center_pos[1] - (bounds[1] + bounds[3]) * 0.5,
        ];
        Ok(rects
            .into_iter()
            .map(|(rect, glyph)| {
                TextInstance::new(
                    [rect[0] + shift[0], rect[1] + shift[1]],
                    [rect[2] + shift[0], rect[3] + shift[1]],
                    [glyph.uv[0], glyph.uv[1]],
                    [glyph.uv[2], glyph.uv[3]],
                    color,
                )
                .with_halo(halo_color, halo_width)
            })
            .collect())
    }

    /// Build rotated GPU quads for a shaped stream already placed along a path.
    ///
    /// The placement slice corresponds to the full visual glyph stream, including
    /// zero-advance attached marks and whitespace. Only outlined glyphs emit quads.
    pub fn layout_shaped_on_placements(
        &self,
        shaped: &ShapedText,
        line_ranges: &[Range<usize>],
        placements: &[GlyphPlacement],
        color: [f32; 4],
        halo_color: [f32; 4],
        halo_width: f32,
    ) -> Result<Vec<TextInstance>, String> {
        let positioned =
            positioned_glyphs(shaped, line_ranges).map_err(|error| error.to_string())?;
        if positioned.len() != placements.len() {
            return Err(format!(
                "Shaped glyph/placement count mismatch: {} glyphs, {} placements",
                positioned.len(),
                placements.len()
            ));
        }
        let scale = shaped.size / self.atlas_font_size;
        let mut line_pens = HashMap::<usize, f32>::new();
        positioned
            .into_iter()
            .zip(placements)
            .filter_map(|(glyph, placement)| {
                let pen = *line_pens.entry(glyph.line_index).or_insert(0.0);
                *line_pens.get_mut(&glyph.line_index).unwrap() += glyph.advance[0];
                glyph.path.is_some().then_some((glyph, placement, pen))
            })
            .map(|(glyph, placement, pen)| {
                let metric = self
                    .get_glyph_id(glyph.font_index, glyph.glyph_id)
                    .ok_or_else(|| {
                        format!(
                            "Atlas is missing shaped glyph {}:{}",
                            glyph.font_index, glyph.glyph_id
                        )
                    })?;
                let width = metric.width * scale;
                let height = metric.height * scale;
                // `placement.screen_pos` is the path sample at pen+advance/2.
                // Preserve GPOS offsets and atlas bearings exactly once by
                // moving the quad center relative to that sample, then rotate
                // the offset into the path tangent frame.
                let center = path_glyph_center(&glyph, metric, placement, pen, scale, shaped.size);
                let half_width = width * 0.5;
                let half_height = height * 0.5;
                let mut instance = TextInstance::new(
                    [center[0] - half_width, center[1] - half_height],
                    [center[0] + half_width, center[1] + half_height],
                    [metric.uv[0], metric.uv[1]],
                    [metric.uv[2], metric.uv[3]],
                    color,
                )
                .with_halo(halo_color, halo_width);
                instance.rotation = placement.rotation;
                Ok(instance)
            })
            .collect()
    }

    /// Measure text dimensions at a given size.
    /// Returns (width, height) in pixels.
    pub fn measure_text(&self, text: &str, size: f32) -> (f32, f32) {
        let scale = size / self.atlas_font_size;
        let mut width = 0.0f32;
        let mut max_height = 0.0f32;

        for c in text.chars() {
            if let Some(glyph) = self.get_glyph(c) {
                width += glyph.advance * scale;
                max_height = max_height.max(glyph.height * scale);
            } else if c == ' ' {
                // Space fallback
                width += size * 0.3;
            }
        }

        (width, max_height.max(size))
    }

    /// Layout text into TextInstance quads.
    /// Generates one SDF/MSDF instance per glyph; the shader expands halos.
    pub fn layout_text(
        &self,
        text: &str,
        center_pos: [f32; 2],
        size: f32,
        color: [f32; 4],
        halo_color: [f32; 4],
        halo_width: f32,
    ) -> Vec<TextInstance> {
        let mut instances = Vec::new();
        let scale = size / self.atlas_font_size;

        // Measure total width to center
        let (total_width, total_height) = self.measure_text(text, size);
        let start_x = center_pos[0] - total_width * 0.5;
        let start_y = center_pos[1] - total_height * 0.5;

        let mut cursor_x = start_x;
        for c in text.chars() {
            if let Some(glyph) = self.get_glyph(c) {
                let x0 = cursor_x + glyph.offset_x * scale;
                let y0 = start_y + glyph.offset_y * scale;
                let x1 = x0 + glyph.width * scale;
                let y1 = y0 + glyph.height * scale;

                instances.push(
                    TextInstance::new(
                        [x0, y0],
                        [x1, y1],
                        [glyph.uv[0], glyph.uv[1]],
                        [glyph.uv[2], glyph.uv[3]],
                        color,
                    )
                    .with_halo(halo_color, halo_width),
                );

                cursor_x += glyph.advance * scale;
            } else if c == ' ' {
                cursor_x += size * 0.3;
            }
        }

        instances
    }
}

impl Clone for MsdfAtlas {
    fn clone(&self) -> Self {
        Self {
            texture: Arc::clone(&self.texture),
            view: Arc::clone(&self.view),
            width: self.width,
            height: self.height,
            glyphs: self.glyphs.clone(),
            unicode_map: self.unicode_map.clone(),
            atlas_font_size: self.atlas_font_size,
            line_height: self.line_height,
            baseline: self.baseline,
            channels: self.channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{path_glyph_center, GlyphKey, GlyphMetrics, MsdfAtlas};
    use crate::core::text_overlay::TextInstance;
    use crate::labels::positioned::PositionedGlyph;
    use crate::labels::renderer_channels_from_atlas;
    use crate::labels::types::GlyphPlacement;

    #[test]
    fn parses_metrics_with_serde_json() {
        let json = r#"{
            "font_size": 24,
            "line_height": 32,
            "baseline": 20,
            "glyphs": {
                "65": {"x": 4, "y": 8, "w": 16, "h": 18, "ox": -2, "oy": 1, "adv": 17}
            }
        }"#;

        let (glyphs, unicode_map, font_size, line_height, baseline, channels) =
            MsdfAtlas::parse_metrics(json, 64, 64).expect("metrics should parse");
        let key = unicode_map[&65];
        let glyph = glyphs.get(&key).expect("A glyph should be present");

        assert_eq!(font_size, 24.0);
        assert_eq!(line_height, 32.0);
        assert_eq!(baseline, 20.0);
        assert_eq!(channels, 1);
        assert_eq!(glyph.width, 16.0);
        assert_eq!(glyph.height, 18.0);
        assert_eq!(glyph.advance, 17.0);
        assert_eq!(glyph.uv, [4.0 / 64.0, 8.0 / 64.0, 20.0 / 64.0, 26.0 / 64.0]);
    }

    #[test]
    fn rejects_empty_or_malformed_metrics() {
        assert!(MsdfAtlas::parse_metrics(r#"{"glyphs": {}}"#, 64, 64).is_err());
        assert!(MsdfAtlas::parse_metrics(
            r#"{"font_size": 12, "line_height": 16, "baseline": 12, "glyphs": {"A": {"x": 0, "y": 0, "w": 1, "h": 1}}}"#,
            64,
            64
        )
        .is_err());
    }

    #[test]
    fn atlas_metadata_drives_rgb_renderer_mode() {
        let json = r#"{
            "channels": 3,
            "glyphs": {"65": {"x": 0, "y": 0, "w": 8, "h": 8, "adv": 7}}
        }"#;
        let (_, _, _, _, _, channels) = MsdfAtlas::parse_metrics(json, 8, 8).unwrap();
        assert_eq!(renderer_channels_from_atlas(channels), 3);
    }

    #[test]
    fn canonical_metrics_are_keyed_by_font_and_shaped_glyph_id() {
        let json = r#"{
            "channels": 3,
            "glyphs": {
                "102": {"x": 0, "y": 0, "w": 8, "h": 8, "font_index": 1, "glyph_id": 700}
            },
            "glyphs_by_id": {
                "1:700": {"x": 0, "y": 0, "w": 8, "h": 8, "font_index": 1, "glyph_id": 700}
            },
            "unicode_map": {"102": "1:700"}
        }"#;
        let (glyphs, unicode_map, ..) = MsdfAtlas::parse_metrics(json, 8, 8).unwrap();
        let key = GlyphKey {
            font_index: 1,
            glyph_id: 700,
        };
        assert_eq!(unicode_map[&102], key);
        assert_eq!(glyphs[&key].font_index, 1);
        assert_eq!(glyphs[&key].glyph_id, 700);
    }

    #[test]
    fn path_layout_rotates_gpos_origin_and_atlas_bearing_exactly_once() {
        let glyph = PositionedGlyph {
            glyph_id: 7,
            font_index: 0,
            cluster: 0,
            line_index: 0,
            origin: [18.0, -4.0],
            advance: [10.0, 0.0],
            path: None,
        };
        let metric = GlyphMetrics {
            codepoint: 0,
            font_index: 0,
            glyph_id: 7,
            uv: [0.0, 0.0, 1.0, 1.0],
            width: 8.0,
            height: 12.0,
            offset_x: 2.0,
            offset_y: 3.0,
            advance: 10.0,
        };
        let placement = GlyphPlacement {
            screen_pos: [100.0, 50.0],
            rotation: std::f32::consts::FRAC_PI_2,
            scale: 20.0,
        };

        let center = path_glyph_center(&glyph, &metric, &placement, 10.0, 2.0, 20.0);

        assert!((center[0] - 86.0).abs() < 1.0e-5);
        assert!((center[1] - 65.0).abs() < 1.0e-5);

        // The collision adapter consumes the exact quad produced from these
        // glyph-specific atlas dimensions and the GPOS/bearing-adjusted center.
        let mut instance = TextInstance::new(
            [center[0] - metric.width, center[1] - metric.height],
            [center[0] + metric.width, center[1] + metric.height],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0; 4],
        );
        instance.rotation = placement.rotation;
        let candidate = crate::labels::optimal::candidate_from_line_instances(
            1,
            0,
            std::slice::from_ref(&instance),
            1,
        )
        .expect("authoritative atlas quad is valid");
        assert_eq!(candidate.bounds, [74.0, 57.0, 98.0, 73.0]);
    }
}
