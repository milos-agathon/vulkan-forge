// src/geo/projections/geocentric.rs
// Geographic ⇄ Geocentric (ECEF) conversion (EPSG method 9602), full f64.
// This is what replaces the `as f32` truncation formerly in src/tiles3d/bounds.rs.
// RELEVANT FILES: src/geo/projections/mod.rs, src/tiles3d/bounds.rs, src/camera/anchor.rs

use glam::DVec3;

use super::{Ellipsoid, ProjError, ProjResult, WGS84};

/// Geodetic (degrees, ellipsoidal height metres) → geocentric ECEF metres.
pub fn geodetic_to_ecef(
    ellipsoid: &Ellipsoid,
    lon_deg: f64,
    lat_deg: f64,
    h_m: f64,
) -> ProjResult<DVec3> {
    if !(-90.0..=90.0).contains(&lat_deg) || !lon_deg.is_finite() || !h_m.is_finite() {
        return Err(ProjError::Domain(format!(
            "geodetic input out of range: lon={lon_deg}, lat={lat_deg}, h={h_m}"
        )));
    }
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let nu = ellipsoid.prime_vertical(lat);
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    Ok(DVec3::new(
        (nu + h_m) * cos_lat * cos_lon,
        (nu + h_m) * cos_lat * sin_lon,
        (nu * (1.0 - ellipsoid.e2()) + h_m) * sin_lat,
    ))
}

/// Geocentric ECEF metres → geodetic (lon degrees, lat degrees, ellipsoidal
/// height metres). Bowring's first approximation refined by fixed-point
/// iteration to machine precision (≪ 1e-9 m everywhere on and near the surface).
pub fn ecef_to_geodetic(ellipsoid: &Ellipsoid, ecef: DVec3) -> ProjResult<(f64, f64, f64)> {
    if !ecef.is_finite() {
        return Err(ProjError::Domain("ECEF input must be finite".to_string()));
    }
    let e2 = ellipsoid.e2();
    let p = ecef.x.hypot(ecef.y);
    if p == 0.0 {
        // On the polar axis the longitude is arbitrary; use 0.
        let lat = if ecef.z >= 0.0 { 90.0 } else { -90.0 };
        let h = ecef.z.abs() - ellipsoid.b();
        return Ok((0.0, lat, h));
    }
    let lon = ecef.y.atan2(ecef.x);

    // Bowring's initial parametric-latitude guess.
    let ep2 = ellipsoid.ep2();
    let b = ellipsoid.b();
    let theta = (ecef.z * ellipsoid.a).atan2(p * b);
    let (st, ct) = theta.sin_cos();
    let mut lat = (ecef.z + ep2 * b * st * st * st).atan2(p - e2 * ellipsoid.a * ct * ct * ct);

    // Fixed-point refinement: lat = atan2(z + e² ν sinlat, p).
    for _ in 0..12 {
        let nu = ellipsoid.prime_vertical(lat);
        let next = (ecef.z + e2 * nu * lat.sin()).atan2(p);
        let delta = (next - lat).abs();
        lat = next;
        if delta < 1e-16 {
            break;
        }
    }
    let nu = ellipsoid.prime_vertical(lat);
    // Height: use the numerically dominant axis to stay stable at the poles.
    let h = if lat.abs() > core::f64::consts::FRAC_PI_4 {
        ecef.z / lat.sin() - nu * (1.0 - e2)
    } else {
        p / lat.cos() - nu
    };
    Ok((lon.to_degrees(), lat.to_degrees(), h))
}

fn planetocentric_surface_radius(ellipsoid: &Ellipsoid, lat: f64) -> f64 {
    let (sin_lat, cos_lat) = lat.sin_cos();
    1.0 / (cos_lat * cos_lat / ellipsoid.a.powi(2) + sin_lat * sin_lat / ellipsoid.b().powi(2))
        .sqrt()
}

/// Planetocentric +East longitude/latitude and radial height → body-fixed XYZ.
pub fn planetocentric_to_ecef(
    ellipsoid: &Ellipsoid,
    lon_deg: f64,
    lat_deg: f64,
    h_m: f64,
) -> ProjResult<DVec3> {
    if !lon_deg.is_finite() || !(-90.0..=90.0).contains(&lat_deg) || !h_m.is_finite() {
        return Err(ProjError::Domain(format!(
            "planetocentric input out of range: lon={lon_deg}, lat={lat_deg}, h={h_m}"
        )));
    }
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let radius = planetocentric_surface_radius(ellipsoid, lat) + h_m;
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    Ok(DVec3::new(
        radius * cos_lat * cos_lon,
        radius * cos_lat * sin_lon,
        radius * sin_lat,
    ))
}

/// Body-fixed XYZ → planetocentric +East longitude/latitude and radial height.
pub fn ecef_to_planetocentric(ellipsoid: &Ellipsoid, ecef: DVec3) -> ProjResult<(f64, f64, f64)> {
    if !ecef.is_finite() || ecef.length_squared() == 0.0 {
        return Err(ProjError::Domain(
            "body-fixed input must be finite and non-zero".to_string(),
        ));
    }
    let horizontal = ecef.x.hypot(ecef.y);
    let lat = ecef.z.atan2(horizontal);
    let lon = if horizontal == 0.0 {
        0.0
    } else {
        ecef.y.atan2(ecef.x).to_degrees()
    };
    let height = ecef.length() - planetocentric_surface_radius(ellipsoid, lat);
    Ok((lon, lat.to_degrees(), height))
}

/// WGS84 convenience wrappers (the common case across the tree).
pub fn wgs84_geodetic_to_ecef(lon_deg: f64, lat_deg: f64, h_m: f64) -> ProjResult<DVec3> {
    geodetic_to_ecef(&WGS84, lon_deg, lat_deg, h_m)
}

pub fn wgs84_ecef_to_geodetic(ecef: DVec3) -> ProjResult<(f64, f64, f64)> {
    ecef_to_geodetic(&WGS84, ecef)
}

#[cfg(test)]
mod tests {
    use super::super::dms;
    use super::*;

    /// EPSG Guidance Note 7-2, method 9602 worked example (WGS 84):
    /// X = 3771793.968, Y = 140253.342, Z = 5124304.349
    /// ⇄ φ = 53°48'33.820"N, λ = 2°07'46.380"E, h = 73.0 m.
    /// GN7-2 publishes the geocentric→geographic direction.
    #[test]
    fn epsg_g7_2_worked_example_inverse_within_1mm() {
        let (lon, lat, h) =
            wgs84_ecef_to_geodetic(DVec3::new(3_771_793.968, 140_253.342, 5_124_304.349)).unwrap();
        // Tighten the published worked example to a 1 mm linear equivalent.
        assert!(
            (lat - dms(53.0, 48.0, 33.820)).abs() < 9e-9,
            "lat residual {lat}"
        );
        assert!(
            (lon - dms(2.0, 7.0, 46.380)).abs() < 9e-9,
            "lon residual {lon}"
        );
        assert!((h - 73.0).abs() < 1e-3, "height residual {h}");
    }

    #[test]
    fn epsg_g7_2_worked_example_forward_within_1mm() {
        // Forward direction with the full-precision geodetic result of the
        // inverse must reproduce the published XYZ to well under 1 mm.
        let xyz = DVec3::new(3_771_793.968, 140_253.342, 5_124_304.349);
        let (lon, lat, h) = wgs84_ecef_to_geodetic(xyz).unwrap();
        let back = wgs84_geodetic_to_ecef(lon, lat, h).unwrap();
        assert!(
            (back - xyz).length() < 1e-6,
            "ECEF roundtrip residual {} m",
            (back - xyz).length()
        );
    }

    #[test]
    fn roundtrip_is_machine_precision_over_the_globe() {
        let mut worst = 0.0f64;
        for lat in [-89.9, -60.0, -30.0, 0.0, 15.0, 45.0, 75.0, 89.9] {
            for lon in [-179.5, -90.0, 0.0, 44.0, 120.0, 179.5] {
                for h in [-100.0, 0.0, 8848.0] {
                    let ecef = wgs84_geodetic_to_ecef(lon, lat, h).unwrap();
                    let (lon2, lat2, h2) = wgs84_ecef_to_geodetic(ecef).unwrap();
                    let back = wgs84_geodetic_to_ecef(lon2, lat2, h2).unwrap();
                    worst = worst.max((back - ecef).length());
                }
            }
        }
        // ~2e-9 m is one ulp at Earth-radius magnitudes — machine precision.
        assert!(worst < 5e-9, "worst ECEF roundtrip residual {worst} m");
    }

    #[test]
    fn planetary_planetocentric_grid_closes_below_nanodegree() {
        for body in [&crate::geo::body::MOON, &crate::geo::body::MARS] {
            let mut worst_deg = 0.0f64;
            for row in 0..25 {
                let lat = -88.0 + 176.0 * row as f64 / 24.0;
                for col in 0..40 {
                    let lon = -175.5 + 351.0 * col as f64 / 39.0;
                    let xyz = planetocentric_to_ecef(&body.ellipsoid, lon, lat, 1234.5).unwrap();
                    let (lon2, lat2, height2) =
                        ecef_to_planetocentric(&body.ellipsoid, xyz).unwrap();
                    let dlon = ((lon2 - lon + 180.0).rem_euclid(360.0) - 180.0).abs();
                    worst_deg = worst_deg.max(dlon.max((lat2 - lat).abs()));
                    assert!((height2 - 1234.5).abs() < 1e-8);
                }
            }
            assert!(
                worst_deg < 1e-9,
                "{} planetocentric closure {worst_deg} deg",
                body.name
            );
        }
    }
}
