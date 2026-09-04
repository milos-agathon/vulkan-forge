// src/geo/mod.rs
// Geographic utilities including CRS reprojection
// RELEVANT FILES: src/geo/reproject.rs, python/forge3d/crs.py

pub mod body;
pub mod geodesic;
pub mod geoid;
pub mod projections;
pub mod refraction;
pub mod reproject;
pub mod solar;
mod solar_coefficients;
pub mod units;

// Re-export main types and functions
pub use reproject::GeoError;

#[cfg(feature = "proj")]
pub use reproject::reproject_coords;

/// Check if the proj feature is available
pub fn proj_available() -> bool {
    cfg!(feature = "proj")
}
