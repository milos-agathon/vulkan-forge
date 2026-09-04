//! B13/B14: Terrain analysis - slope/aspect computation and contour extraction
//!
//! This module provides CPU-based analysis functions for terrain data including
//! slope and aspect calculation via finite differences and contour line extraction
//! using marching squares algorithm.

use std::f32::consts::PI;

#[path = "analysis/viewshed.rs"]
pub mod viewshed;

/// Slope and aspect data for a single point
#[derive(Debug, Clone)]
pub struct SlopeAspect {
    /// Slope in degrees (0-90)
    pub slope_deg: f32,
    /// Aspect in degrees (0-360, where 0/360=North, 90=East, 180=South, 270=West)
    pub aspect_deg: f32,
}

/// A polyline representing a contour line
#[derive(Debug, Clone)]
pub struct ContourPolyline {
    /// Level (elevation) this contour represents
    pub level: f32,
    /// Sequence of (x, y) points defining the contour line
    pub points: Vec<(f32, f32)>,
}

/// Contour extraction result containing all contours for requested levels
#[derive(Debug, Clone)]
pub struct ContourResult {
    /// All extracted contour polylines
    pub polylines: Vec<ContourPolyline>,
    /// Total number of polylines extracted
    pub polyline_count: usize,
    /// Total number of points across all polylines
    pub total_points: usize,
}

/// Compute slope and aspect for a height field using finite differences
///
/// # Arguments
/// * `heights` - Height data in row-major order (height[y * width + x])
/// * `width` - Width of the height field in samples
/// * `height` - Height of the height field in samples  
/// * `dx` - Spacing between samples in X direction (world units)
/// * `dy` - Spacing between samples in Y direction (world units)
///
/// # Returns
/// Vector of SlopeAspect values in same row-major order as input heights
///
/// # B13: Acceptance criteria
/// Must match CPU finite-difference reference with RMSE ≤ 0.5° on synthetic ramps/gaussians
pub fn slope_aspect_compute(
    heights: &[f32],
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
) -> Result<Vec<SlopeAspect>, String> {
    if heights.len() != width * height {
        return Err(format!(
            "Height array length {} does not match dimensions {}x{}",
            heights.len(),
            width,
            height
        ));
    }

    if width < 3 || height < 3 {
        return Err("Width and height must be at least 3 for finite differences".to_string());
    }

    if dx <= 0.0 || dy <= 0.0 {
        return Err("dx and dy must be positive".to_string());
    }

    let mut result = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let slope_aspect = compute_slope_aspect_at_point(heights, width, height, x, y, dx, dy);
            result.push(slope_aspect);
        }
    }

    Ok(result)
}

/// Compute slope and aspect at a specific point using finite differences
fn compute_slope_aspect_at_point(
    heights: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    dx: f32,
    dy: f32,
) -> SlopeAspect {
    // Handle boundary conditions by clamping indices
    let x_prev = if x > 0 { x - 1 } else { x };
    let x_next = if x + 1 < width { x + 1 } else { x };
    let y_prev = if y > 0 { y - 1 } else { y };
    let y_next = if y + 1 < height { y + 1 } else { y };

    // Get height values
    let h_left = heights[y * width + x_prev];
    let h_right = heights[y * width + x_next];
    let h_bottom = heights[y_prev * width + x];
    let h_top = heights[y_next * width + x];

    // Calculate gradients using finite differences
    // dz/dx (positive eastward)
    let dz_dx = (h_right - h_left) / (dx * (x_next - x_prev) as f32);
    // dz/dy (positive northward)
    let dz_dy = (h_top - h_bottom) / (dy * (y_next - y_prev) as f32);

    // Calculate slope magnitude in radians, then convert to degrees
    let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
    let slope_deg = slope_rad * 180.0 / PI;

    // Calculate aspect (direction of steepest descent)
    // atan2(-dz_dy, -dz_dx) gives direction of steepest descent
    // We want aspect in geographic convention: 0°=North, 90°=East, etc.
    let aspect_rad = if dz_dx == 0.0 && dz_dy == 0.0 {
        0.0 // Flat area, aspect undefined - default to North
    } else {
        // atan2 returns [-π, π], convert to [0, 2π] then to degrees
        let raw_aspect = (-dz_dy).atan2(-dz_dx);
        let aspect_geographic = raw_aspect + PI / 2.0; // Rotate so 0 = North
        if aspect_geographic < 0.0 {
            aspect_geographic + 2.0 * PI
        } else if aspect_geographic >= 2.0 * PI {
            aspect_geographic - 2.0 * PI
        } else {
            aspect_geographic
        }
    };

    let aspect_deg = aspect_rad * 180.0 / PI;

    SlopeAspect {
        slope_deg: slope_deg.max(0.0).min(90.0), // Clamp to valid range
        aspect_deg: aspect_deg.max(0.0).min(360.0), // Clamp to valid range
    }
}

/// Extract contour lines from height field using marching squares algorithm
///
/// # Arguments
/// * `heights` - Height data in row-major order (height[y * width + x])
/// * `width` - Width of the height field in samples
/// * `height` - Height of the height field in samples
/// * `dx` - Spacing between samples in X direction (world units)
/// * `dy` - Spacing between samples in Y direction (world units)
/// * `levels` - Contour levels to extract
///
/// # Returns
/// ContourResult containing all extracted polylines
///
/// # B14: Acceptance criteria
/// Must return deterministic polyline counts/lengths for level sets on
/// plane/ramp/gaussian DEMs within ±1% tolerance
pub fn contour_extract(
    heights: &[f32],
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
    levels: &[f32],
) -> Result<ContourResult, String> {
    if heights.len() != width * height {
        return Err(format!(
            "Height array length {} does not match dimensions {}x{}",
            heights.len(),
            width,
            height
        ));
    }

    if width < 2 || height < 2 {
        return Err("Width and height must be at least 2 for contouring".to_string());
    }

    if dx <= 0.0 || dy <= 0.0 {
        return Err("dx and dy must be positive".to_string());
    }

    let mut all_polylines = Vec::new();

    // Extract contours for each level
    for &level in levels {
        let polylines = extract_contours_for_level(heights, width, height, dx, dy, level);
        all_polylines.extend(polylines);
    }

    let polyline_count = all_polylines.len();
    let total_points: usize = all_polylines.iter().map(|p| p.points.len()).sum();

    Ok(ContourResult {
        polylines: all_polylines,
        polyline_count,
        total_points,
    })
}

/// Extract contours for a single level using marching squares
fn extract_contours_for_level(
    heights: &[f32],
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
    level: f32,
) -> Vec<ContourPolyline> {
    let mut polylines = Vec::new();

    // Process each cell in the grid
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            // Get the four corner heights of this cell
            let h00 = heights[y * width + x]; // Bottom-left
            let h10 = heights[y * width + (x + 1)]; // Bottom-right
            let h01 = heights[(y + 1) * width + x]; // Top-left
            let h11 = heights[(y + 1) * width + (x + 1)]; // Top-right

            // Determine marching squares case
            let case = compute_marching_squares_case(h00, h10, h01, h11, level);

            if case == 0 || case == 15 {
                continue; // No contour in this cell
            }

            // Extract line segments for this case
            let segments = get_marching_squares_segments(
                case,
                h00,
                h10,
                h01,
                h11,
                level,
                x as f32 * dx,
                y as f32 * dy,
                dx,
                dy,
            );

            // Convert segments to polylines (simplified - each segment becomes a polyline)
            for segment in segments {
                polylines.push(ContourPolyline {
                    level,
                    points: vec![segment.0, segment.1],
                });
            }
        }
    }

    polylines
}

/// Compute marching squares case number (0-15) for a cell
fn compute_marching_squares_case(h00: f32, h10: f32, h01: f32, h11: f32, level: f32) -> u8 {
    let mut case = 0;

    if h00 >= level {
        case |= 1;
    } // Bottom-left
    if h10 >= level {
        case |= 2;
    } // Bottom-right
    if h01 >= level {
        case |= 4;
    } // Top-left
    if h11 >= level {
        case |= 8;
    } // Top-right

    case
}

/// Get line segments for a marching squares case
/// Returns vector of ((x1,y1), (x2,y2)) line segments
fn get_marching_squares_segments(
    case: u8,
    h00: f32,
    h10: f32,
    h01: f32,
    h11: f32,
    level: f32,
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> Vec<((f32, f32), (f32, f32))> {
    // Edge interpolation function
    let interpolate = |h1: f32, h2: f32| -> f32 {
        if (h2 - h1).abs() < 1e-6 {
            0.5 // Avoid division by zero
        } else {
            (level - h1) / (h2 - h1)
        }
    };

    // Edge midpoints with interpolation
    let bottom_t = interpolate(h00, h10);
    let right_t = interpolate(h10, h11);
    let top_t = interpolate(h01, h11);
    let left_t = interpolate(h00, h01);

    // Edge points in world coordinates
    let bottom_pt = (x0 + bottom_t * dx, y0);
    let right_pt = (x0 + dx, y0 + right_t * dy);
    let top_pt = (x0 + top_t * dx, y0 + dy);
    let left_pt = (x0, y0 + left_t * dy);

    // Return line segments based on marching squares case
    match case {
        1 => vec![(left_pt, bottom_pt)],
        2 => vec![(bottom_pt, right_pt)],
        3 => vec![(left_pt, right_pt)],
        4 => vec![(top_pt, left_pt)],
        5 => vec![(top_pt, bottom_pt)],
        6 => vec![(bottom_pt, right_pt), (top_pt, left_pt)],
        7 => vec![(top_pt, right_pt)],
        8 => vec![(right_pt, top_pt)],
        9 => vec![(left_pt, bottom_pt), (right_pt, top_pt)],
        10 => vec![(bottom_pt, top_pt)],
        11 => vec![(left_pt, top_pt)],
        12 => vec![(right_pt, left_pt)],
        13 => vec![(right_pt, bottom_pt)],
        14 => vec![(bottom_pt, left_pt)],
        _ => vec![], // Cases 0 and 15 have no contours
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slope_aspect_flat_surface() {
        // Flat surface should have zero slope
        let heights = vec![10.0; 9]; // 3x3 grid, all same height
        let result = slope_aspect_compute(&heights, 3, 3, 1.0, 1.0).unwrap();

        // All points should have near-zero slope
        for sa in result.iter() {
            assert!(
                sa.slope_deg < 0.1,
                "Slope should be near zero for flat surface, got {}",
                sa.slope_deg
            );
        }
    }

    #[test]
    fn test_slope_aspect_simple_ramp() {
        // Simple east-facing ramp: heights increase from west to east
        let heights = vec![
            0.0, 1.0, 2.0, // Bottom row
            0.0, 1.0, 2.0, // Middle row
            0.0, 1.0, 2.0, // Top row
        ];

        let result = slope_aspect_compute(&heights, 3, 3, 1.0, 1.0).unwrap();

        // Center point should have a meaningful slope magnitude.
        let center = &result[4]; // Middle point
        assert!(
            center.slope_deg > 30.0,
            "Ramp should have significant slope"
        );

        // Aspect is reported as the direction of steepest descent.
        // With heights increasing west->east, downhill points west.
        assert!(
            center.aspect_deg > 225.0 && center.aspect_deg < 315.0,
            "Westward downslope expected for this ramp, got {}",
            center.aspect_deg
        );
    }

    #[test]
    fn test_contour_extraction_basic() {
        // Simple height field with clear level crossings
        let heights = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let levels = vec![1.0, 2.0, 3.0];
        let result = contour_extract(&heights, 3, 3, 1.0, 1.0, &levels).unwrap();

        // Should find some contour lines
        assert!(
            result.polyline_count > 0,
            "Should extract some contour lines"
        );
        assert!(result.total_points > 0, "Should have some contour points");

        // All polylines should have the requested levels
        for polyline in &result.polylines {
            assert!(
                levels.contains(&polyline.level),
                "Polyline level should be one of the requested levels"
            );
        }
    }

    #[test]
    fn test_contour_extraction_no_crossings() {
        // Flat surface - no level crossings
        let heights = vec![5.0; 9]; // 3x3 grid, all height 5.0
        let levels = vec![1.0, 2.0, 3.0]; // Levels below the surface

        let result = contour_extract(&heights, 3, 3, 1.0, 1.0, &levels).unwrap();

        // Should find no contour lines
        assert_eq!(
            result.polyline_count, 0,
            "Flat surface should produce no contours"
        );
    }

    #[test]
    fn test_slope_aspect_input_validation() {
        // Test error conditions
        let heights = vec![1.0, 2.0, 3.0, 4.0];

        // Wrong array size
        assert!(slope_aspect_compute(&heights, 3, 3, 1.0, 1.0).is_err());

        // Too small dimensions
        assert!(slope_aspect_compute(&heights, 2, 2, 1.0, 1.0).is_err());

        // Invalid spacing
        assert!(slope_aspect_compute(&heights, 2, 2, 0.0, 1.0).is_err());
        assert!(slope_aspect_compute(&heights, 2, 2, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_contour_extraction_input_validation() {
        let heights = vec![1.0, 2.0, 3.0, 4.0];
        let levels = vec![2.5];

        // Wrong array size
        assert!(contour_extract(&heights, 3, 3, 1.0, 1.0, &levels).is_err());

        // Too small dimensions
        assert!(contour_extract(&[1.0], 1, 1, 1.0, 1.0, &levels).is_err());

        // Invalid spacing
        assert!(contour_extract(&heights, 2, 2, 0.0, 1.0, &levels).is_err());
    }
}
