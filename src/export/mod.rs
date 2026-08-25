//! Vector export module for SVG/PDF generation.
//!
//! Provides functionality to export vector graphics (polygons, polylines, labels)
//! to SVG format for print-grade overlays.
//!
//! # Features
//! - 3D to 2D projection with view-projection matrix
//! - 2D bounds to screen coordinate mapping
//! - SVG generation with polygon and polyline elements
//! - Label text rendering with halo support

pub(crate) mod projection;
mod svg;
mod svg_labels;

pub use projection::{
    compute_bounds_from_geometry, project_2d_points_to_screen, project_2d_to_screen,
    project_3d_to_2d, Bounds2D,
};
pub use svg::{vectors_to_svg, vectors_to_svg_screen_coords, SvgExportConfig};
pub use svg_labels::{
    label_at_position, labels_to_svg_document, labels_to_svg_text, labels_to_svg_text_with_config,
    try_label_at_position, try_labels_to_svg_document, try_labels_to_svg_text_with_config,
    LabelSvgConfig, SvgTextError,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::{LabelData, LabelId, LabelStyle};
    use crate::vector::api::{PolygonDef, PolylineDef, VectorStyle};
    use glam::{Mat4, Vec2, Vec3};

    #[test]
    fn test_project_3d_to_2d_center() {
        // Identity view-proj puts (0,0,0) at center of screen
        let view_proj = Mat4::IDENTITY;
        let viewport = (800, 600);
        let result = project_3d_to_2d(Vec3::ZERO, &view_proj, viewport);
        assert!(result.is_some());
        let (x, y) = result.unwrap();
        assert!((x - 400.0).abs() < 0.01);
        assert!((y - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_project_3d_behind_camera() {
        // Point behind camera (positive Z in right-handed coords with identity matrix)
        // After projection, w <= 0 means behind camera
        let view_proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let viewport = (800, 600);
        // Point behind camera (z = 10, but camera looks toward -Z)
        let result = project_3d_to_2d(Vec3::new(0.0, 0.0, 10.0), &view_proj, viewport);
        // In RH perspective, points at positive Z are behind the camera
        assert!(result.is_none());
    }

    #[test]
    fn test_project_2d_to_screen() {
        let bounds = Bounds2D {
            min: Vec2::new(0.0, 0.0),
            max: Vec2::new(100.0, 100.0),
        };
        let viewport = (800, 600);

        // Bottom-left corner
        let (x, y) = project_2d_to_screen(Vec2::new(0.0, 0.0), &bounds, viewport);
        assert!((x - 0.0).abs() < 0.01);
        assert!((y - 600.0).abs() < 0.01); // Y flipped

        // Top-right corner
        let (x, y) = project_2d_to_screen(Vec2::new(100.0, 100.0), &bounds, viewport);
        assert!((x - 800.0).abs() < 0.01);
        assert!((y - 0.0).abs() < 0.01); // Y flipped

        // Center
        let (x, y) = project_2d_to_screen(Vec2::new(50.0, 50.0), &bounds, viewport);
        assert!((x - 400.0).abs() < 0.01);
        assert!((y - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_svg_generation() {
        let polygons = vec![PolygonDef {
            exterior: vec![
                Vec2::new(10.0, 10.0),
                Vec2::new(90.0, 10.0),
                Vec2::new(50.0, 90.0),
            ],
            holes: vec![],
            style: VectorStyle {
                fill_color: [1.0, 0.0, 0.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 2.0,
                point_size: 4.0,
            },
        }];

        let polylines = vec![PolylineDef {
            path: vec![Vec2::new(0.0, 50.0), Vec2::new(100.0, 50.0)],
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: [0.0, 0.0, 1.0, 1.0],
                stroke_width: 1.5,
                point_size: 4.0,
            },
        }];

        let bounds = Bounds2D {
            min: Vec2::new(0.0, 0.0),
            max: Vec2::new(100.0, 100.0),
        };

        let svg = vectors_to_svg(
            &polygons,
            &polylines,
            &bounds,
            800,
            600,
            &SvgExportConfig::default(),
        );

        // Verify SVG structure
        assert!(svg.contains("<?xml version"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("viewBox=\"0 0 800 600\""));
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_labels_to_svg() {
        let labels = vec![LabelData {
            id: LabelId(1),
            text: "Test Label".to_string(),
            world_pos: glam::DVec3::new(50.0, 0.0, 50.0),
            render_pos: Vec3::new(50.0, 0.0, 50.0),
            style: LabelStyle {
                size: 14.0,
                color: [0.0, 0.0, 0.0, 1.0],
                halo_color: [1.0, 1.0, 1.0, 0.8],
                halo_width: 1.5,
                ..Default::default()
            },
            screen_pos: Some([400.0, 300.0]),
            visible: true,
            depth: 0.5,
            horizon_angle: 45.0,
            computed_alpha: 1.0,
        }];

        let svg_text = labels_to_svg_text(&labels);

        assert!(!svg_text.contains("<text"));
        assert!(svg_text.contains("<path"));
        assert!(svg_text.contains("d=\"M"));
    }
}
