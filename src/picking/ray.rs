// src/picking/ray.rs
// Ray unprojection utilities for GPU ray picking
// Part of Plan 2: Standard - GPU Ray Picking + Hover Support

/// A ray in 3D space defined by an origin and direction
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self { origin, direction }
    }

    /// Get a point along the ray at parameter t
    pub fn point_at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + self.direction[0] * t,
            self.origin[1] + self.direction[1] * t,
            self.origin[2] + self.direction[2] * t,
        ]
    }
}

/// Unproject a screen coordinate to a world-space ray
///
/// # Arguments
/// * `screen_x` - X coordinate in pixels (0 = left)
/// * `screen_y` - Y coordinate in pixels (0 = top)
/// * `screen_width` - Screen width in pixels
/// * `screen_height` - Screen height in pixels
/// * `inv_view_proj` - Inverse of the combined view-projection matrix
///
/// # Returns
/// A Ray from the camera through the given screen point
pub fn unproject_cursor(
    screen_x: u32,
    screen_y: u32,
    screen_width: u32,
    screen_height: u32,
    inv_view_proj: [[f32; 4]; 4],
) -> Ray {
    // Address the centre of the raster pixel. The GPU visibility buffer is
    // evaluated at pixel centres, so unprojecting the integer-valued top-left
    // corner gives a different triangle/background answer along every edge.
    let ndc_x = (2.0 * (screen_x as f32 + 0.5) / screen_width as f32) - 1.0;
    let ndc_y = 1.0 - (2.0 * (screen_y as f32 + 0.5) / screen_height as f32); // Y is flipped

    // Near and far points in NDC
    let near_ndc = [ndc_x, ndc_y, 0.0, 1.0];
    let far_ndc = [ndc_x, ndc_y, 1.0, 1.0];

    // Transform to world space
    let near_world = transform_point(near_ndc, inv_view_proj);
    let far_world = transform_point(far_ndc, inv_view_proj);

    // Compute direction
    let direction = normalize([
        far_world[0] - near_world[0],
        far_world[1] - near_world[1],
        far_world[2] - near_world[2],
    ]);

    Ray {
        origin: near_world,
        direction,
    }
}

/// Transform a homogeneous point by a 4x4 matrix and perform perspective divide
fn transform_point(point: [f32; 4], matrix: [[f32; 4]; 4]) -> [f32; 3] {
    let x = matrix[0][0] * point[0]
        + matrix[1][0] * point[1]
        + matrix[2][0] * point[2]
        + matrix[3][0] * point[3];
    let y = matrix[0][1] * point[0]
        + matrix[1][1] * point[1]
        + matrix[2][1] * point[2]
        + matrix[3][1] * point[3];
    let z = matrix[0][2] * point[0]
        + matrix[1][2] * point[1]
        + matrix[2][2] * point[2]
        + matrix[3][2] * point[3];
    let w = matrix[0][3] * point[0]
        + matrix[1][3] * point[1]
        + matrix[2][3] * point[2]
        + matrix[3][3] * point[3];

    if w.abs() < 1e-10 {
        [x, y, z]
    } else {
        [x / w, y / w, z / w]
    }
}

/// Normalize a 3D vector
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Invert a 4x4 matrix
/// Returns None if the matrix is singular
pub fn invert_matrix(m: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] =
        m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2] - m[2][1] * m[1][2] * m[3][3]
            + m[2][1] * m[1][3] * m[3][2]
            + m[3][1] * m[1][2] * m[2][3]
            - m[3][1] * m[1][3] * m[2][2];

    inv[1][0] =
        -m[1][0] * m[2][2] * m[3][3] + m[1][0] * m[2][3] * m[3][2] + m[2][0] * m[1][2] * m[3][3]
            - m[2][0] * m[1][3] * m[3][2]
            - m[3][0] * m[1][2] * m[2][3]
            + m[3][0] * m[1][3] * m[2][2];

    inv[2][0] =
        m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1] - m[2][0] * m[1][1] * m[3][3]
            + m[2][0] * m[1][3] * m[3][1]
            + m[3][0] * m[1][1] * m[2][3]
            - m[3][0] * m[1][3] * m[2][1];

    inv[3][0] =
        -m[1][0] * m[2][1] * m[3][2] + m[1][0] * m[2][2] * m[3][1] + m[2][0] * m[1][1] * m[3][2]
            - m[2][0] * m[1][2] * m[3][1]
            - m[3][0] * m[1][1] * m[2][2]
            + m[3][0] * m[1][2] * m[2][1];

    inv[0][1] =
        -m[0][1] * m[2][2] * m[3][3] + m[0][1] * m[2][3] * m[3][2] + m[2][1] * m[0][2] * m[3][3]
            - m[2][1] * m[0][3] * m[3][2]
            - m[3][1] * m[0][2] * m[2][3]
            + m[3][1] * m[0][3] * m[2][2];

    inv[1][1] =
        m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2] - m[2][0] * m[0][2] * m[3][3]
            + m[2][0] * m[0][3] * m[3][2]
            + m[3][0] * m[0][2] * m[2][3]
            - m[3][0] * m[0][3] * m[2][2];

    inv[2][1] =
        -m[0][0] * m[2][1] * m[3][3] + m[0][0] * m[2][3] * m[3][1] + m[2][0] * m[0][1] * m[3][3]
            - m[2][0] * m[0][3] * m[3][1]
            - m[3][0] * m[0][1] * m[2][3]
            + m[3][0] * m[0][3] * m[2][1];

    inv[3][1] =
        m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1] - m[2][0] * m[0][1] * m[3][2]
            + m[2][0] * m[0][2] * m[3][1]
            + m[3][0] * m[0][1] * m[2][2]
            - m[3][0] * m[0][2] * m[2][1];

    inv[0][2] =
        m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2] - m[1][1] * m[0][2] * m[3][3]
            + m[1][1] * m[0][3] * m[3][2]
            + m[3][1] * m[0][2] * m[1][3]
            - m[3][1] * m[0][3] * m[1][2];

    inv[1][2] =
        -m[0][0] * m[1][2] * m[3][3] + m[0][0] * m[1][3] * m[3][2] + m[1][0] * m[0][2] * m[3][3]
            - m[1][0] * m[0][3] * m[3][2]
            - m[3][0] * m[0][2] * m[1][3]
            + m[3][0] * m[0][3] * m[1][2];

    inv[2][2] =
        m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1] - m[1][0] * m[0][1] * m[3][3]
            + m[1][0] * m[0][3] * m[3][1]
            + m[3][0] * m[0][1] * m[1][3]
            - m[3][0] * m[0][3] * m[1][1];

    inv[3][2] =
        -m[0][0] * m[1][1] * m[3][2] + m[0][0] * m[1][2] * m[3][1] + m[1][0] * m[0][1] * m[3][2]
            - m[1][0] * m[0][2] * m[3][1]
            - m[3][0] * m[0][1] * m[1][2]
            + m[3][0] * m[0][2] * m[1][1];

    inv[0][3] =
        -m[0][1] * m[1][2] * m[2][3] + m[0][1] * m[1][3] * m[2][2] + m[1][1] * m[0][2] * m[2][3]
            - m[1][1] * m[0][3] * m[2][2]
            - m[2][1] * m[0][2] * m[1][3]
            + m[2][1] * m[0][3] * m[1][2];

    inv[1][3] =
        m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2] - m[1][0] * m[0][2] * m[2][3]
            + m[1][0] * m[0][3] * m[2][2]
            + m[2][0] * m[0][2] * m[1][3]
            - m[2][0] * m[0][3] * m[1][2];

    inv[2][3] =
        -m[0][0] * m[1][1] * m[2][3] + m[0][0] * m[1][3] * m[2][1] + m[1][0] * m[0][1] * m[2][3]
            - m[1][0] * m[0][3] * m[2][1]
            - m[2][0] * m[0][1] * m[1][3]
            + m[2][0] * m[0][3] * m[1][1];

    inv[3][3] =
        m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] - m[1][0] * m[0][1] * m[2][2]
            + m[1][0] * m[0][2] * m[2][1]
            + m[2][0] * m[0][1] * m[1][2]
            - m[2][0] * m[0][2] * m[1][1];

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

    if det.abs() < 1e-10 {
        return None;
    }

    let inv_det = 1.0 / det;
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] *= inv_det;
        }
    }

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_point_at() {
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let point = ray.point_at(5.0);
        assert!((point[0] - 5.0).abs() < 1e-6);
        assert!(point[1].abs() < 1e-6);
        assert!(point[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = normalize([3.0, 4.0, 0.0]);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unproject_cursor_addresses_the_raster_pixel_center() {
        let ray = unproject_cursor(0, 0, 2, 2, glam::Mat4::IDENTITY.to_cols_array_2d());
        assert_eq!(ray.origin, [-0.5, 0.5, 0.0]);
        assert_eq!(ray.direction, [0.0, 0.0, 1.0]);
    }
}
