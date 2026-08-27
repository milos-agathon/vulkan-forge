//! Orbit camera helpers for the terrain renderer.
//!
//! Computes camera transforms shared between Python and Rust layers.
use glam::{Mat4, Vec3};

/// Calculate an orbit camera position around a target.
///
/// # Arguments
/// - `target`: Center point to orbit around.
/// - `radius`: Distance from target (must be positive and finite).
/// - `phi_deg`: Azimuth angle in degrees (rotation around Y axis).
/// - `theta_deg`: Polar angle in degrees (elevation from vertical).
pub fn orbit_camera(target: Vec3, radius: f32, phi_deg: f32, theta_deg: f32) -> Vec3 {
    if !radius.is_finite() || radius <= 0.0 {
        return target;
    }

    let phi_rad = phi_deg.to_radians();
    let theta_rad = theta_deg.to_radians();

    // Spherical to Cartesian conversion (right-handed, Y up).
    let x = radius * theta_rad.sin() * phi_rad.cos();
    let y = radius * theta_rad.cos();
    let z = radius * theta_rad.sin() * phi_rad.sin();

    target + Vec3::new(x, y, z)
}

/// Build the view-projection matrices for the terrain camera.
///
/// Returns `(view_matrix, projection_matrix)` for right-handed Y-up coordinates.
pub fn build_view_proj(
    eye: Vec3,
    target: Vec3,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (Mat4, Mat4) {
    let up = Vec3::Y;

    let view = Mat4::look_at_rh(eye, target, up);
    let proj = crate::camera::perspective_wgpu(fov_y_deg.to_radians(), aspect, near, far);
    (view, proj)
}

/// Build the canonical Y-up orbit camera used by both terrain presentation and
/// NEPHELE's reference-camera adapter.  Keeping the orbit calculation here
/// prevents separately rounded eye/basis values from becoming fixture inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_orbit_view_proj(
    target: Vec3,
    radius: f32,
    phi_deg: f32,
    theta_deg: f32,
    fov_y_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (Vec3, Mat4, Mat4) {
    let eye = orbit_camera(target, radius, phi_deg, theta_deg);
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    // Match TerrainScene's established WGPU projection bit-for-bit; the older
    // public build_view_proj helper retains its existing implementation.
    let projection = Mat4::perspective_rh(fov_y_deg.to_radians(), aspect, near, far);
    (eye, view, projection)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracked_nephele_orbit_eye_and_basis_are_bitwise_stable() {
        let target = Vec3::new(0.0, 15.0, 0.0);
        let (eye, view, _) =
            build_orbit_view_proj(target, 50.990_195, 90.0, 78.690_07, 45.0, 1.0, 0.1, 1_000.0);
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let forward = (target - eye).normalize();

        assert_eq!(
            eye.to_array().map(f32::to_bits),
            [0xb612_abcc, 0x41c7_ffff, 0x4248_0000]
        );
        assert_eq!(
            right.to_array().map(f32::to_bits),
            [0x3f80_0000, 0x8000_0000, 0x333b_bd2e]
        );
        assert_eq!(
            up.to_array().map(f32::to_bits),
            [0x3213_4649, 0x3f7b_0756, 0xbe48_d2a9]
        );
        assert_eq!(
            forward.to_array().map(f32::to_bits),
            [0x3338_17dd, 0xbe48_d2a9, 0xbf7b_0756]
        );
    }
}
