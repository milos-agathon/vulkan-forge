//! Curved text layout along Bézier/polyline paths.
//!
//! Provides per-glyph positioning with tangent-following rotation
//! for atlas-style curved text rendering.

use glam::{Mat4, Vec2, Vec3};

/// A point along a path with its tangent direction.
#[derive(Debug, Clone, Copy)]
pub struct PathPoint {
    /// Position in world space.
    pub position: Vec3,
    /// Tangent direction (normalized).
    pub tangent: Vec3,
    /// Distance along the path from start.
    pub distance: f32,
}

/// Sampled path with precomputed distances and tangents.
#[derive(Debug, Clone)]
pub struct SampledPath {
    /// Path points with positions, tangents, and distances.
    pub points: Vec<PathPoint>,
    /// Total arc length of the path.
    pub total_length: f32,
}

impl SampledPath {
    /// Create a sampled path from a polyline.
    pub fn from_polyline(vertices: &[Vec3]) -> Self {
        if vertices.len() < 2 {
            return Self {
                points: Vec::new(),
                total_length: 0.0,
            };
        }

        let mut points = Vec::with_capacity(vertices.len());
        let mut total_dist = 0.0;

        for i in 0..vertices.len() {
            let pos = vertices[i];

            // Compute tangent
            let tangent = if i == 0 {
                (vertices[1] - vertices[0]).normalize_or_zero()
            } else if i == vertices.len() - 1 {
                (vertices[i] - vertices[i - 1]).normalize_or_zero()
            } else {
                // Average of incoming and outgoing tangents
                let t1 = (vertices[i] - vertices[i - 1]).normalize_or_zero();
                let t2 = (vertices[i + 1] - vertices[i]).normalize_or_zero();
                ((t1 + t2) * 0.5).normalize_or_zero()
            };

            points.push(PathPoint {
                position: pos,
                tangent,
                distance: total_dist,
            });

            if i < vertices.len() - 1 {
                total_dist += (vertices[i + 1] - vertices[i]).length();
            }
        }

        Self {
            points,
            total_length: total_dist,
        }
    }

    /// Sample the path at a given arc-length distance.
    pub fn sample_at_distance(&self, distance: f32) -> Option<PathPoint> {
        if self.points.is_empty() || distance < 0.0 || distance > self.total_length {
            return None;
        }

        // Find the segment containing this distance
        for i in 0..self.points.len() - 1 {
            let p0 = &self.points[i];
            let p1 = &self.points[i + 1];

            if distance >= p0.distance && distance <= p1.distance {
                let segment_len = p1.distance - p0.distance;
                if segment_len < 0.0001 {
                    return Some(*p0);
                }

                let t = (distance - p0.distance) / segment_len;
                let position = p0.position.lerp(p1.position, t);
                let tangent = p0.tangent.lerp(p1.tangent, t).normalize_or_zero();

                return Some(PathPoint {
                    position,
                    tangent,
                    distance,
                });
            }
        }

        // Return last point if at end
        self.points.last().copied()
    }

    /// Check if the path is long enough for the given text width.
    pub fn can_fit_text(&self, text_width: f32) -> bool {
        self.total_length >= text_width
    }
}

/// Glyph instance for curved text rendering.
#[derive(Debug, Clone, Copy)]
pub struct CurvedGlyphInstance {
    /// World position of the glyph center.
    pub world_pos: Vec3,
    /// Rotation angle in radians (from tangent).
    pub rotation: f32,
    /// UV rectangle in atlas: [u_min, v_min, u_max, v_max].
    pub uv_rect: [f32; 4],
    /// Glyph color.
    pub color: [f32; 4],
    /// Scale factor.
    pub scale: f32,
    /// Distance along path where this glyph is placed.
    pub path_offset: f32,
    /// Character this glyph represents.
    pub character: char,
}

/// Layout result for curved text.
#[derive(Debug, Clone)]
pub struct CurvedTextLayout {
    /// Per-glyph instances.
    pub glyphs: Vec<CurvedGlyphInstance>,
    /// Total width of the laid out text.
    pub total_width: f32,
    /// Whether the text was successfully placed.
    pub success: bool,
}

/// Screen-space glyph geometry emitted by the curved-layout projection
/// authority. Solver adapters consume this output without recomputing layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedCurvedGlyph {
    pub screen_pos: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
}

/// Compute curved text layout along a path.
///
/// Places each glyph at uniform arc-length intervals along the path,
/// with rotation following the path tangent.
pub fn layout_curved_text(
    text: &str,
    path: &SampledPath,
    glyph_advances: &[f32],
    font_size: f32,
    color: [f32; 4],
    tracking: f32,
    center_on_path: bool,
) -> CurvedTextLayout {
    if text.is_empty() || path.points.is_empty() || glyph_advances.is_empty() {
        return CurvedTextLayout {
            glyphs: Vec::new(),
            total_width: 0.0,
            success: false,
        };
    }

    // Calculate total text width with tracking
    let chars: Vec<char> = text.chars().collect();
    let mut total_width = 0.0;
    for (i, _) in chars.iter().enumerate() {
        if i < glyph_advances.len() {
            total_width += glyph_advances[i] * font_size;
            if i < chars.len() - 1 {
                total_width += tracking * font_size;
            }
        }
    }

    // Check if path can fit the text
    if !path.can_fit_text(total_width) {
        return CurvedTextLayout {
            glyphs: Vec::new(),
            total_width,
            success: false,
        };
    }

    // Calculate starting offset (center text on path if requested)
    let start_offset = if center_on_path {
        (path.total_length - total_width) / 2.0
    } else {
        0.0
    };

    let mut glyphs = Vec::with_capacity(chars.len());
    let mut current_offset = start_offset;

    for (i, &ch) in chars.iter().enumerate() {
        let advance = if i < glyph_advances.len() {
            glyph_advances[i] * font_size
        } else {
            font_size * 0.5 // Fallback
        };

        // Sample path at glyph center
        let glyph_center_offset = current_offset + advance * 0.5;
        if let Some(point) = path.sample_at_distance(glyph_center_offset) {
            // Calculate rotation from tangent (2D rotation in screen space)
            let rotation = point.tangent.x.atan2(point.tangent.z);

            glyphs.push(CurvedGlyphInstance {
                world_pos: point.position,
                rotation,
                uv_rect: [0.0, 0.0, 1.0, 1.0], // Filled by atlas after packing.
                color,
                scale: font_size,
                path_offset: glyph_center_offset,
                character: ch,
            });
        }

        current_offset += advance;
        if i < chars.len() - 1 {
            current_offset += tracking * font_size;
        }
    }

    CurvedTextLayout {
        glyphs,
        total_width,
        success: true,
    }
}

/// Project curved text layout to screen space.
pub fn project_curved_glyphs(
    layout: &CurvedTextLayout,
    view_proj: Mat4,
    screen_width: f32,
    screen_height: f32,
) -> Vec<ProjectedCurvedGlyph> {
    let mut screen_positions = Vec::with_capacity(layout.glyphs.len());

    for glyph in &layout.glyphs {
        let clip = view_proj * glyph.world_pos.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }

        let ndc = clip.truncate() / clip.w;
        let screen_x = (ndc.x * 0.5 + 0.5) * screen_width;
        let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * screen_height;

        // Project tangent to get screen-space rotation
        // `layout_curved_text` records atan2(tangent.x, tangent.z), so invert
        // that convention exactly when projecting the tangent endpoint.
        let tangent_world = Vec3::new(glyph.rotation.sin(), 0.0, glyph.rotation.cos());
        let tangent_end = glyph.world_pos + tangent_world * 10.0;
        let clip_end = view_proj * tangent_end.extend(1.0);

        if clip_end.w <= 0.0 {
            continue;
        }
        let ndc_end = clip_end.truncate() / clip_end.w;
        let screen_end_x = (ndc_end.x * 0.5 + 0.5) * screen_width;
        let screen_end_y = (1.0 - (ndc_end.y * 0.5 + 0.5)) * screen_height;
        let screen_rotation = (screen_end_y - screen_y).atan2(screen_end_x - screen_x);

        screen_positions.push(ProjectedCurvedGlyph {
            screen_pos: [screen_x, screen_y],
            rotation: screen_rotation,
            scale: glyph.scale,
        });
    }

    screen_positions
}

/// Compatibility projection surface retaining the historical tuple shape.
pub fn project_curved_layout(
    layout: &CurvedTextLayout,
    view_proj: Mat4,
    screen_width: f32,
    screen_height: f32,
) -> Vec<(Vec2, f32)> {
    project_curved_glyphs(layout, view_proj, screen_width, screen_height)
        .into_iter()
        .map(|glyph| {
            (
                Vec2::new(glyph.screen_pos[0], glyph.screen_pos[1]),
                glyph.rotation,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampled_path_from_polyline() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
        ];
        let path = SampledPath::from_polyline(&vertices);

        assert_eq!(path.points.len(), 3);
        assert!((path.total_length - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_at_distance() {
        let vertices = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let path = SampledPath::from_polyline(&vertices);

        let mid = path.sample_at_distance(5.0).unwrap();
        assert!((mid.position.x - 5.0).abs() < 0.001);
    }
}
