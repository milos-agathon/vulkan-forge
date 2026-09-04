//! WGS84 curvature and atmospheric-refraction models shared by long-range analyses.

const WGS84_A_M: f64 = 6_378_137.0;
const WGS84_E2: f64 = 6.694_379_990_141_316_5e-3;

pub fn principal_radii_m(latitude_deg: f64) -> Result<(f64, f64), String> {
    if !latitude_deg.is_finite() || !(-90.0..=90.0).contains(&latitude_deg) {
        return Err("latitude must be finite and in [-90, 90]".into());
    }
    let phi = latitude_deg.to_radians();
    let w = (1.0 - WGS84_E2 * phi.sin().powi(2)).sqrt();
    Ok((WGS84_A_M * (1.0 - WGS84_E2) / w.powi(3), WGS84_A_M / w))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EarthModel {
    /// Disable the vertical curvature drop. Geographic analyses still use
    /// WGS84 geodesics horizontally so flat-vs-curved ablations change one
    /// physical variable instead of also changing the DEM distance metric.
    Flat,
    Sphere {
        radius_m: f64,
    },
    Ellipsoid {
        latitude_deg: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefractionModel {
    None,
    Bennett {
        pressure_mbar: f64,
        temperature_c: f64,
    },
    Saemundsson {
        pressure_mbar: f64,
        temperature_c: f64,
    },
    EffectiveRadius {
        k: f64,
    },
}

impl EarthModel {
    pub fn from_name(name: &str, latitude_deg: f64, sphere_radius_m: f64) -> Result<Self, String> {
        match name {
            "flat" => Ok(Self::Flat),
            "sphere" => Ok(Self::Sphere {
                radius_m: sphere_radius_m,
            }),
            "ellipsoid" | "wgs84" => Ok(Self::Ellipsoid { latitude_deg }),
            _ => Err(format!("unsupported earth_model {name:?}")),
        }
    }

    pub fn directional_radius_m(self, azimuth_deg: f64) -> Result<f64, String> {
        if !azimuth_deg.is_finite() {
            return Err("azimuth must be finite".into());
        }
        match self {
            Self::Flat => Ok(f64::INFINITY),
            Self::Sphere { radius_m } if radius_m.is_finite() && radius_m > 0.0 => Ok(radius_m),
            Self::Sphere { .. } => Err("sphere radius must be finite and positive".into()),
            Self::Ellipsoid { latitude_deg }
                if latitude_deg.is_finite() && (-90.0..=90.0).contains(&latitude_deg) =>
            {
                let (meridional, prime_vertical) = principal_radii_m(latitude_deg)?;
                let azimuth = azimuth_deg.to_radians();
                Ok(1.0
                    / (azimuth.cos().powi(2) / meridional + azimuth.sin().powi(2) / prime_vertical))
            }
            Self::Ellipsoid { .. } => Err("latitude must be finite and in [-90, 90]".into()),
        }
    }
}

impl RefractionModel {
    pub fn from_name(
        name: &str,
        pressure_mbar: f64,
        temperature_c: f64,
        k: f64,
    ) -> Result<Self, String> {
        match name {
            "none" => Ok(Self::None),
            "bennett" => Ok(Self::Bennett {
                pressure_mbar,
                temperature_c,
            }),
            "saemundsson" => Ok(Self::Saemundsson {
                pressure_mbar,
                temperature_c,
            }),
            "effective_radius" => Ok(Self::EffectiveRadius { k }),
            _ => Err(format!("unsupported refraction_model {name:?}")),
        }
    }

    pub fn k(self) -> Result<f64, String> {
        let value = match self {
            Self::None => 0.0,
            Self::EffectiveRadius { k } => k,
            Self::Bennett {
                pressure_mbar,
                temperature_c,
            } => standard_k(pressure_mbar, temperature_c, 0.13)?,
            Self::Saemundsson {
                pressure_mbar,
                temperature_c,
            } => standard_k(pressure_mbar, temperature_c, 1.0 / 7.0)?,
        };
        if value.is_finite() && value < 1.0 {
            Ok(value)
        } else {
            Err("refraction k must be finite and less than 1".into())
        }
    }
}

pub fn effective_radius_m(
    earth: EarthModel,
    refraction: RefractionModel,
    azimuth_deg: f64,
) -> Result<f64, String> {
    if matches!(earth, EarthModel::Flat) && !matches!(refraction, RefractionModel::None) {
        return Err("flat earth only supports refraction_model='none'".into());
    }
    Ok(earth.directional_radius_m(azimuth_deg)? / (1.0 - refraction.k()?))
}

pub fn curvature_drop_m(distance_m: f64, effective_radius_m: f64) -> Result<f64, String> {
    if !distance_m.is_finite() || distance_m < 0.0 || effective_radius_m <= 0.0 {
        return Err("distance must be finite/non-negative and radius positive".into());
    }
    Ok(distance_m * distance_m / (2.0 * effective_radius_m))
}

fn standard_k(pressure_mbar: f64, temperature_c: f64, base: f64) -> Result<f64, String> {
    if !pressure_mbar.is_finite() || pressure_mbar <= 0.0 || temperature_c <= -273.15 {
        return Err("pressure must be positive and temperature above absolute zero".into());
    }
    Ok(base * (pressure_mbar / 1013.25) * (288.15 / (273.15 + temperature_c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgs84_directional_radius_uses_meridian_and_prime_vertical() {
        let earth = EarthModel::Ellipsoid { latitude_deg: 45.0 };
        assert!(
            earth.directional_radius_m(90.0).unwrap() > earth.directional_radius_m(0.0).unwrap()
        );
        let effective =
            effective_radius_m(earth, RefractionModel::EffectiveRadius { k: 0.13 }, 0.0).unwrap();
        assert!(effective > earth.directional_radius_m(0.0).unwrap());
    }

    #[test]
    fn model_names_are_strict_and_flat_still_validates_refraction() {
        assert!(EarthModel::from_name("mean-earth", 0.0, 6_371_000.0).is_err());
        assert!(RefractionModel::from_name("standard", 1013.25, 15.0, 0.13).is_err());
        assert!(effective_radius_m(
            EarthModel::Flat,
            RefractionModel::EffectiveRadius { k: 1.0 },
            0.0
        )
        .is_err());
        assert!(effective_radius_m(
            EarthModel::Flat,
            RefractionModel::Bennett {
                pressure_mbar: 1013.25,
                temperature_c: 15.0,
            },
            0.0,
        )
        .is_err());
        assert!(
            effective_radius_m(EarthModel::Flat, RefractionModel::None, 0.0)
                .unwrap()
                .is_infinite()
        );
    }
}
