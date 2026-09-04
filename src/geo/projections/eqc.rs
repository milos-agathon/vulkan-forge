// Equidistant Cylindrical (Equirectangular), EPSG methods 1028/1029.
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

use super::{Ellipsoid, ProjError, ProjResult};

/// Equidistant Cylindrical projection on a sphere or ellipsoid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equirectangular {
    pub ellipsoid: Ellipsoid,
    pub lat0_deg: f64,
    pub lon0_deg: f64,
    pub lat_ts_deg: f64,
    pub false_easting: f64,
    pub false_northing: f64,
}

impl Equirectangular {
    fn meridional_arc(&self, lat: f64) -> f64 {
        let e2 = self.ellipsoid.e2();
        let e4 = e2 * e2;
        let e6 = e4 * e2;
        let e8 = e4 * e4;
        self.ellipsoid.a
            * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0 - 175.0 * e8 / 16_384.0) * lat
                - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1_024.0 + 105.0 * e8 / 4_096.0)
                    * (2.0 * lat).sin()
                + (15.0 * e4 / 256.0 + 45.0 * e6 / 1_024.0 + 525.0 * e8 / 16_384.0)
                    * (4.0 * lat).sin()
                - (35.0 * e6 / 3_072.0 + 175.0 * e8 / 12_288.0) * (6.0 * lat).sin()
                + 315.0 * e8 / 131_072.0 * (8.0 * lat).sin())
    }

    fn latitude_from_arc(&self, arc: f64) -> ProjResult<f64> {
        let mut lat = arc / self.ellipsoid.a;
        for _ in 0..8 {
            let sin_lat = lat.sin();
            let derivative = self.ellipsoid.a * (1.0 - self.ellipsoid.e2())
                / (1.0 - self.ellipsoid.e2() * sin_lat * sin_lat).powf(1.5);
            let delta = (self.meridional_arc(lat) - arc) / derivative;
            lat -= delta;
            if delta.abs() < 1e-15 {
                return Ok(lat);
            }
        }
        Err(ProjError::Convergence(
            "equirectangular inverse latitude did not converge".to_string(),
        ))
    }

    fn x_scale(&self) -> ProjResult<f64> {
        let lat_ts = self.lat_ts_deg.to_radians();
        let scale = self.ellipsoid.prime_vertical(lat_ts) * lat_ts.cos();
        if scale.abs() <= f64::EPSILON {
            return Err(ProjError::Domain(
                "equirectangular latitude of true scale must not be a pole".to_string(),
            ));
        }
        Ok(scale)
    }

    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        if !lon_deg.is_finite() || !(-90.0..=90.0).contains(&lat_deg) {
            return Err(ProjError::Domain(format!(
                "equirectangular input out of range: lon={lon_deg}, lat={lat_deg}"
            )));
        }
        let delta_lon = (lon_deg - self.lon0_deg + 180.0).rem_euclid(360.0) - 180.0;
        Ok((
            self.false_easting + self.x_scale()? * delta_lon.to_radians(),
            self.false_northing + self.meridional_arc(lat_deg.to_radians())
                - self.meridional_arc(self.lat0_deg.to_radians()),
        ))
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        if !easting.is_finite() || !northing.is_finite() {
            return Err(ProjError::Domain(
                "equirectangular inverse input must be finite".to_string(),
            ));
        }
        let lon = self.lon0_deg + ((easting - self.false_easting) / self.x_scale()?).to_degrees();
        let arc = northing - self.false_northing + self.meridional_arc(self.lat0_deg.to_radians());
        let lat = self.latitude_from_arc(arc)?.to_degrees();
        Ok(((lon + 180.0).rem_euclid(360.0) - 180.0, lat))
    }
}

#[cfg(test)]
mod tests {
    use super::super::WGS84;
    use super::*;

    #[test]
    fn proj_documented_wgs84_example() {
        let projection = Equirectangular {
            ellipsoid: WGS84,
            lat0_deg: 0.0,
            lon0_deg: 0.0,
            lat_ts_deg: 0.0,
            false_easting: 0.0,
            false_northing: 0.0,
        };
        let (x, y) = projection.forward(2.0, 47.0).unwrap();
        assert!((x - 222_638.98).abs() < 0.01, "x={x}");
        assert!((y - 5_207_247.01).abs() < 0.01, "y={y}");
        let (lon, lat) = projection.inverse(x, y).unwrap();
        assert!((lon - 2.0).abs() < 1e-12);
        assert!((lat - 47.0).abs() < 1e-10);
    }
}
