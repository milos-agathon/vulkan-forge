//! Planetary datum registry and compile-time body markers.
//!
//! Shape and rotational constants follow the IAU WGCCRE 2015 report
//! (Archinal et al., 2018, doi:10.1007/s10569-017-9805-5). `prime_meridian_w0`
//! is degrees at J2000 and `rotation_rate` is degrees per day. The Moon and
//! Mars values are the constant terms of the report's orientation models;
//! periodic orientation terms are outside SELENE's datum-only scope.

use thiserror::Error;

use super::projections::{Ellipsoid, WGS84};

/// Gravity surface attached to a registered body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GravitySurfaceId {
    Egm96,
    MarsAreoid,
}

impl GravitySurfaceId {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Egm96 => "EGM96",
            Self::MarsAreoid => "Mars areoid",
        }
    }
}

/// Runtime description of a planetary datum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub name: &'static str,
    pub ellipsoid: Ellipsoid,
    pub prime_meridian_w0: f64,
    pub rotation_rate: f64,
    pub gravity_surface: Option<GravitySurfaceId>,
}

/// WGS84 Earth. WGCCRE 2015 delegates Earth orientation to IERS; the
/// conventional J2000 mean-angle fields are retained as registry metadata.
pub const EARTH: Body = Body {
    name: "Earth",
    ellipsoid: WGS84,
    prime_meridian_w0: 190.147,
    rotation_rate: 360.985_623_5,
    gravity_surface: Some(GravitySurfaceId::Egm96),
};

/// IAU mean-Earth/polar-axis Moon; cartographic reference radius 1737.4 km
/// (WGCCRE 2015, Table 5).
pub const MOON: Body = Body {
    name: "Moon",
    ellipsoid: Ellipsoid {
        a: 1_737_400.0,
        f: 0.0,
    },
    prime_meridian_w0: 38.3213,
    rotation_rate: 13.176_358_15,
    gravity_surface: None,
};

/// IAU 2000 Mars reference ellipsoid and WGCCRE 2015 orientation constants.
pub const MARS: Body = Body {
    name: "Mars",
    ellipsoid: Ellipsoid::new(3_396_190.0, 169.894_447_223_612),
    prime_meridian_w0: 176.630,
    rotation_rate: 350.891_982_26,
    gravity_surface: Some(GravitySurfaceId::MarsAreoid),
};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("unsupported body '{name}'; expected earth, moon, or mars")]
pub struct BodyError {
    name: String,
}

/// Resolve a body name without silently substituting Earth.
pub fn body(name: &str) -> Result<&'static Body, BodyError> {
    match name.trim().to_ascii_lowercase().as_str() {
        "earth" => Ok(&EARTH),
        "moon" => Ok(&MOON),
        "mars" => Ok(&MARS),
        _ => Err(BodyError {
            name: name.to_string(),
        }),
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Marker implemented only by bodies in the compile-time datum registry.
pub trait BodyTag: sealed::Sealed + Copy + core::fmt::Debug + 'static {
    const BODY: &'static Body;
    const BODY_FIXED_AUTHORITY: &'static str;
    const BODY_FIXED_CODE: u32;
    const BODY_FIXED_NAME: &'static str;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Earth {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moon {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mars {}

impl sealed::Sealed for Earth {}
impl sealed::Sealed for Moon {}
impl sealed::Sealed for Mars {}

impl BodyTag for Earth {
    const BODY: &'static Body = &EARTH;
    const BODY_FIXED_AUTHORITY: &'static str = "EPSG";
    const BODY_FIXED_CODE: u32 = 4978;
    const BODY_FIXED_NAME: &'static str = "Earth-centred, Earth-fixed";
}
impl BodyTag for Moon {
    const BODY: &'static Body = &MOON;
    const BODY_FIXED_AUTHORITY: &'static str = "FORGE3D";
    const BODY_FIXED_CODE: u32 = 301;
    const BODY_FIXED_NAME: &'static str = "Moon-centred, Moon-fixed";
}
impl BodyTag for Mars {
    const BODY: &'static Body = &MARS;
    const BODY_FIXED_AUTHORITY: &'static str = "FORGE3D";
    const BODY_FIXED_CODE: u32 = 499;
    const BODY_FIXED_NAME: &'static str = "Mars-centred, Mars-fixed";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_the_three_supported_bodies_without_fallbacks() {
        assert_eq!(body("earth").unwrap(), &EARTH);
        assert_eq!(body("MOON").unwrap(), &MOON);
        assert_eq!(body("Mars").unwrap(), &MARS);
        assert_eq!(
            body("ceres").unwrap_err().to_string(),
            "unsupported body 'ceres'; expected earth, moon, or mars"
        );
    }

    #[test]
    fn registry_uses_the_required_reference_surfaces() {
        assert_eq!(EARTH.ellipsoid.a, 6_378_137.0);
        assert_eq!(MOON.ellipsoid.a, 1_737_400.0);
        assert_eq!(MOON.ellipsoid.f, 0.0);
        assert_eq!(MARS.ellipsoid.a, 3_396_190.0);
        assert_eq!(1.0 / MARS.ellipsoid.f, 169.894_447_223_612);
        assert_eq!(EARTH.gravity_surface, Some(GravitySurfaceId::Egm96));
        assert_eq!(MOON.gravity_surface, None);
        assert_eq!(MARS.gravity_surface, Some(GravitySurfaceId::MarsAreoid));
    }
}
