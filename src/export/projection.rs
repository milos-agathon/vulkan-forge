//! Coordinate projection utilities for vector export.
//!
//! Handles 3D to 2D projection using view-projection matrices and
//! 2D bounds to screen coordinate mapping.

#[cfg(feature = "extension-module")]
use crate::core::error::{RenderError, RenderResult};
use glam::{Mat4, Vec2, Vec3};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

/// 2D axis-aligned bounding box for coordinate mapping.
#[derive(Debug, Clone, Copy)]
pub struct Bounds2D {
    /// Minimum corner (bottom-left in standard coordinates).
    pub min: Vec2,
    /// Maximum corner (top-right in standard coordinates).
    pub max: Vec2,
}

impl Bounds2D {
    /// Create bounds from min/max points.
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Create bounds from (min_x, min_y, max_x, max_y).
    pub fn from_extents(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min: Vec2::new(min_x, min_y),
            max: Vec2::new(max_x, max_y),
        }
    }

    /// Width of the bounding box.
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Expand bounds to include a point.
    pub fn expand_to_include(&mut self, point: Vec2) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
    }

    /// Create bounds from a set of points.
    pub fn from_points(points: &[Vec2]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut bounds = Self {
            min: points[0],
            max: points[0],
        };

        for point in points.iter().skip(1) {
            bounds.expand_to_include(*point);
        }

        Some(bounds)
    }

    /// Add padding to the bounds.
    pub fn with_padding(&self, padding: f32) -> Self {
        Self {
            min: self.min - Vec2::splat(padding),
            max: self.max + Vec2::splat(padding),
        }
    }
}

impl Default for Bounds2D {
    fn default() -> Self {
        Self {
            min: Vec2::ZERO,
            max: Vec2::ONE,
        }
    }
}

/// Project a 3D point to 2D screen coordinates.
///
/// Uses the view-projection matrix to transform the point and then maps
/// from normalized device coordinates (NDC) to screen coordinates.
///
/// # Arguments
/// * `point` - 3D world position
/// * `view_proj` - Combined view-projection matrix
/// * `viewport` - Screen dimensions (width, height)
///
/// # Returns
/// `Some((x, y))` in screen coordinates, or `None` if the point is behind the camera.
///
/// # Example
/// ```ignore
/// let view_proj = camera.view_proj_matrix();
/// if let Some((x, y)) = project_3d_to_2d(world_pos, &view_proj, (1920, 1080)) {
///     // Point is visible at screen position (x, y)
/// }
/// ```
pub fn project_3d_to_2d(point: Vec3, view_proj: &Mat4, viewport: (u32, u32)) -> Option<(f32, f32)> {
    // Transform to clip space
    let clip = *view_proj * point.extend(1.0);

    // Behind camera check (w <= 0 means point is behind or at the camera plane)
    if clip.w <= 0.0 {
        return None;
    }

    // Perspective division to NDC (Normalized Device Coordinates)
    // NDC range is [-1, 1] for x and y
    let ndc = clip.truncate() / clip.w;

    // Check if outside NDC bounds (optional - could allow for extrapolation)
    // For SVG export, we often want to include off-screen geometry
    // so we don't clip here

    // Map NDC to screen coordinates
    // X: [-1, 1] -> [0, width]
    // Y: [-1, 1] -> [height, 0] (Y is flipped for screen coordinates)
    let x = (ndc.x + 1.0) * 0.5 * viewport.0 as f32;
    let y = (1.0 - ndc.y) * 0.5 * viewport.1 as f32;

    Some((x, y))
}

/// Project a 2D point from bounds coordinates to screen coordinates.
///
/// Maps a point from the source coordinate system (defined by bounds)
/// to screen pixel coordinates.
///
/// # Arguments
/// * `point` - 2D point in source coordinates
/// * `bounds` - Bounding box defining the source coordinate range
/// * `viewport` - Screen dimensions (width, height)
///
/// # Returns
/// Screen coordinates (x, y) where:
/// - X increases left to right
/// - Y increases top to bottom (screen convention)
///
/// # Example
/// ```ignore
/// let bounds = Bounds2D::from_extents(0.0, 0.0, 1000.0, 1000.0);
/// let (screen_x, screen_y) = project_2d_to_screen(
///     Vec2::new(500.0, 500.0),
///     &bounds,
///     (800, 600)
/// );
/// // Result: (400.0, 300.0) - center of screen
/// ```
pub fn project_2d_to_screen(point: Vec2, bounds: &Bounds2D, viewport: (u32, u32)) -> (f32, f32) {
    let range = bounds.max - bounds.min;

    // Avoid division by zero
    let range_x = if range.x.abs() < 1e-10 { 1.0 } else { range.x };
    let range_y = if range.y.abs() < 1e-10 { 1.0 } else { range.y };

    // Normalize point to [0, 1] range within bounds
    let normalized_x = (point.x - bounds.min.x) / range_x;
    let normalized_y = (point.y - bounds.min.y) / range_y;

    // Map to screen coordinates
    // Y is flipped: normalized_y=0 (bottom) -> screen y=height
    //               normalized_y=1 (top)    -> screen y=0
    let x = normalized_x * viewport.0 as f32;
    let y = (1.0 - normalized_y) * viewport.1 as f32;

    (x, y)
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "project_3d_to_2d")]
pub fn project_3d_to_2d_py(
    point: (f32, f32, f32),
    view_proj: Vec<f32>,
    viewport: (u32, u32),
) -> RenderResult<Option<(f32, f32)>> {
    if view_proj.len() != 16 {
        return Err(RenderError::Upload(
            "view_proj must contain 16 column-major values".into(),
        ));
    }
    let matrix = Mat4::from_cols_slice(&view_proj);
    Ok(project_3d_to_2d(
        Vec3::new(point.0, point.1, point.2),
        &matrix,
        viewport,
    ))
}

#[cfg(feature = "extension-module")]
#[pyfunction(name = "project_2d_to_screen")]
pub fn project_2d_to_screen_py(
    point: (f32, f32),
    bounds: (f32, f32, f32, f32),
    viewport: (u32, u32),
) -> (f32, f32) {
    project_2d_to_screen(
        Vec2::new(point.0, point.1),
        &Bounds2D::from_extents(bounds.0, bounds.1, bounds.2, bounds.3),
        viewport,
    )
}

/// Project multiple 2D points from bounds coordinates to screen coordinates.
///
/// Convenience function for batch projection.
pub fn project_2d_points_to_screen(
    points: &[Vec2],
    bounds: &Bounds2D,
    viewport: (u32, u32),
) -> Vec<(f32, f32)> {
    points
        .iter()
        .map(|p| project_2d_to_screen(*p, bounds, viewport))
        .collect()
}

/// Compute bounds from a collection of polygons and polylines.
pub fn compute_bounds_from_geometry(
    polygon_exteriors: &[Vec<Vec2>],
    polyline_paths: &[Vec<Vec2>],
) -> Option<Bounds2D> {
    let mut all_points: Vec<Vec2> = Vec::new();

    for exterior in polygon_exteriors {
        all_points.extend(exterior.iter().cloned());
    }

    for path in polyline_paths {
        all_points.extend(path.iter().cloned());
    }

    Bounds2D::from_points(&all_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_from_extents() {
        let bounds = Bounds2D::from_extents(10.0, 20.0, 110.0, 120.0);
        assert_eq!(bounds.width(), 100.0);
        assert_eq!(bounds.height(), 100.0);
        assert_eq!(bounds.center(), Vec2::new(60.0, 70.0));
    }

    #[test]
    fn test_bounds_expand() {
        let mut bounds = Bounds2D::new(Vec2::ZERO, Vec2::ONE);
        bounds.expand_to_include(Vec2::new(2.0, -1.0));
        assert_eq!(bounds.min, Vec2::new(0.0, -1.0));
        assert_eq!(bounds.max, Vec2::new(2.0, 1.0));
    }

    #[test]
    fn test_bounds_from_points() {
        let points = vec![
            Vec2::new(10.0, 20.0),
            Vec2::new(30.0, 40.0),
            Vec2::new(5.0, 35.0),
        ];
        let bounds = Bounds2D::from_points(&points).unwrap();
        assert_eq!(bounds.min, Vec2::new(5.0, 20.0));
        assert_eq!(bounds.max, Vec2::new(30.0, 40.0));
    }

    #[test]
    fn test_bounds_padding() {
        let bounds = Bounds2D::new(Vec2::new(10.0, 10.0), Vec2::new(90.0, 90.0));
        let padded = bounds.with_padding(5.0);
        assert_eq!(padded.min, Vec2::new(5.0, 5.0));
        assert_eq!(padded.max, Vec2::new(95.0, 95.0));
    }
}
