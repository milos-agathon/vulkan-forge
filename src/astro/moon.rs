//! Meeus-class 60-term lunar longitude, distance, and latitude theory.
//!
//! The complete tables from *Astronomical Algorithms*, chapter 47, are
//! committed in compact form (`assets/astro/moon_terms.bin`). The declared
//! truncation is that of the published tables themselves — nothing further is
//! dropped — and the *measured* maximum topocentric error over the committed
//! 621-epoch, 30-day 2000–2050 Horizons sweep is the figure carried in
//! `assets/astro/MANIFEST.toml`, gated by
//! `tests/test_astro_ephemeris.py::test_manifest_truncation_budgets_are_gates_not_prose`.
//! SIDERA's DoD threshold for the Moon is the looser 30 arcseconds.
//!
//! Note that this module returns the *geometric* geocentric position; the
//! light-time retardation that makes it apparent is applied by
//! [`super::apparent_geocentric_equatorial`], which deliberately does not add
//! annual aberration for the Moon.

use super::AstroError;
use glam::DVec3;
use std::io::{Cursor, Read};
use std::sync::OnceLock;

const DATA: &[u8] = include_bytes!("../../assets/astro/moon_terms.bin");
const AU_KM: f64 = 149_597_870.7;
const J2000: f64 = 2_451_545.0;

#[derive(Clone, Copy, Debug)]
pub struct LunarCoordinates {
    pub ecliptic_of_date_au: DVec3,
    pub distance_km: f64,
}

#[derive(Debug)]
struct Theory {
    longitude_radius: Vec<LongitudeRadiusTerm>,
    latitude: Vec<LatitudeTerm>,
}

#[derive(Debug)]
struct LongitudeRadiusTerm {
    argument: [i8; 4],
    longitude: i32,
    radius: i32,
}

#[derive(Debug)]
struct LatitudeTerm {
    argument: [i8; 4],
    latitude: i32,
}

pub fn geocentric_ecliptic(jd_tt: f64) -> Result<LunarCoordinates, AstroError> {
    let t = (jd_tt - J2000) / 36_525.0;
    let l_prime = polynomial(
        t,
        &[
            218.316_447_7,
            481_267.881_234_21,
            -0.001_578_6,
            1.0 / 538_841.0,
            -1.0 / 65_194_000.0,
        ],
    );
    let d = polynomial(
        t,
        &[
            297.850_192_1,
            445_267.111_403_4,
            -0.001_881_9,
            1.0 / 545_868.0,
            -1.0 / 113_065_000.0,
        ],
    );
    let m = polynomial(
        t,
        &[
            357.529_109_2,
            35_999.050_290_9,
            -0.000_153_6,
            1.0 / 24_490_000.0,
        ],
    );
    let m_prime = polynomial(
        t,
        &[
            134.963_396_4,
            477_198.867_505_5,
            0.008_741_4,
            1.0 / 69_699.0,
            -1.0 / 14_712_000.0,
        ],
    );
    let f = polynomial(
        t,
        &[
            93.272_095,
            483_202.017_523_3,
            -0.003_653_9,
            -1.0 / 3_526_000.0,
            1.0 / 863_310_000.0,
        ],
    );
    let arguments = [d, m, m_prime, f].map(f64::to_radians);
    let eccentricity = 1.0 - 0.002_516 * t - 0.000_007_4 * t * t;
    let theory = theory()?;
    let mut longitude_sum = 0.0;
    let mut radius_sum = 0.0;
    for term in &theory.longitude_radius {
        let argument = dot(term.argument, arguments);
        let scale = eccentricity.powi(i32::from(term.argument[1].unsigned_abs()));
        longitude_sum += f64::from(term.longitude) * scale * argument.sin();
        radius_sum += f64::from(term.radius) * scale * argument.cos();
    }
    let a1 = (119.75 + 131.849 * t).to_radians();
    let a2 = (53.09 + 479_264.290 * t).to_radians();
    longitude_sum +=
        3_958.0 * a1.sin() + 1_962.0 * (l_prime - f).to_radians().sin() + 318.0 * a2.sin();

    let mut latitude_sum = 0.0;
    for term in &theory.latitude {
        let argument = dot(term.argument, arguments);
        let scale = eccentricity.powi(i32::from(term.argument[1].unsigned_abs()));
        latitude_sum += f64::from(term.latitude) * scale * argument.sin();
    }
    let a3 = (313.45 + 481_266.484 * t).to_radians();
    latitude_sum += -2_235.0 * l_prime.to_radians().sin()
        + 382.0 * a3.sin()
        + 175.0 * (a1 - f.to_radians()).sin()
        + 175.0 * (a1 + f.to_radians()).sin()
        + 127.0 * (l_prime - m_prime).to_radians().sin()
        - 115.0 * (l_prime + m_prime).to_radians().sin();

    let longitude = (l_prime + longitude_sum / 1_000_000.0).to_radians();
    let latitude = (latitude_sum / 1_000_000.0).to_radians();
    let distance_km = 385_000.56 + radius_sum / 1_000.0;
    let (sin_l, cos_l) = longitude.sin_cos();
    let (sin_b, cos_b) = latitude.sin_cos();
    Ok(LunarCoordinates {
        ecliptic_of_date_au: DVec3::new(cos_b * cos_l, cos_b * sin_l, sin_b)
            * (distance_km / AU_KM),
        distance_km,
    })
}

fn dot(coefficients: [i8; 4], arguments: [f64; 4]) -> f64 {
    coefficients
        .iter()
        .zip(arguments)
        .map(|(coefficient, argument)| f64::from(*coefficient) * argument)
        .sum()
}

fn polynomial(t: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * t + coefficient)
        .rem_euclid(360.0)
}

fn theory() -> Result<&'static Theory, AstroError> {
    static THEORY: OnceLock<Result<Theory, AstroError>> = OnceLock::new();
    THEORY
        .get_or_init(|| parse_bytes(DATA))
        .as_ref()
        .map_err(Clone::clone)
}

fn parse_bytes(data: &[u8]) -> Result<Theory, AstroError> {
    let mut input = Cursor::new(data);
    let mut magic = [0; 8];
    input
        .read_exact(&mut magic)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    if &magic != b"F3DMOON1" {
        return Err(AstroError::InvalidData("moon_terms.bin"));
    }
    let longitude_count = read_u32(&mut input)? as usize;
    let latitude_count = read_u32(&mut input)? as usize;
    let mut longitude_radius = Vec::with_capacity(longitude_count);
    for _ in 0..longitude_count {
        longitude_radius.push(LongitudeRadiusTerm {
            argument: read_arguments(&mut input)?,
            longitude: read_i32(&mut input)?,
            radius: read_i32(&mut input)?,
        });
    }
    let mut latitude = Vec::with_capacity(latitude_count);
    for _ in 0..latitude_count {
        latitude.push(LatitudeTerm {
            argument: read_arguments(&mut input)?,
            latitude: read_i32(&mut input)?,
        });
    }
    if input.position() != data.len() as u64 {
        return Err(AstroError::InvalidData("moon_terms.bin"));
    }
    Ok(Theory {
        longitude_radius,
        latitude,
    })
}

fn read_arguments(input: &mut Cursor<&[u8]>) -> Result<[i8; 4], AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    Ok(bytes.map(|value| value as i8))
}

fn read_u32(input: &mut Cursor<&[u8]>) -> Result<u32, AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(input: &mut Cursor<&[u8]>) -> Result<i32, AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    Ok(i32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Meeus tables 47.A and 47.B are 60 rows each; the committed asset must
    /// carry both in full and consume every byte.
    #[test]
    fn committed_theory_round_trips_with_the_full_meeus_tables() {
        let theory = theory().expect("committed lunar asset parses");
        assert_eq!(theory.longitude_radius.len(), 60);
        assert_eq!(theory.latitude.len(), 60);
        // Table 47.A row 1: D=0 M=0 M'=1 F=0, Sigma-l 6288774, Sigma-r -20905355.
        let first = &theory.longitude_radius[0];
        assert_eq!(first.argument, [0, 0, 1, 0]);
        assert_eq!((first.longitude, first.radius), (6_288_774, -20_905_355));
        // Table 47.B row 1: D=0 M=0 M'=0 F=1, Sigma-b 5128122.
        let first_latitude = &theory.latitude[0];
        assert_eq!(first_latitude.argument, [0, 0, 0, 1]);
        assert_eq!(first_latitude.latitude, 5_128_122);
        // The eccentricity correction only ever sees |M| <= 2.
        assert!(theory
            .longitude_radius
            .iter()
            .all(|term| term.argument[1].abs() <= 2));
        assert!(theory
            .latitude
            .iter()
            .all(|term| term.argument[1].abs() <= 2));
    }

    /// Meeus example 47.a: 1992 April 12, 0h TD. Published results are
    /// lambda = 133.162655 deg, beta = -3.229126 deg, Delta = 368409.7 km.
    /// The epoch is outside SIDERA's public window on purpose — the kernel
    /// takes a raw Julian date, so the textbook identity is checked where the
    /// textbook states it.
    #[test]
    fn matches_meeus_example_47a() {
        let coordinates = geocentric_ecliptic(2_448_724.5).unwrap();
        let direction = coordinates.ecliptic_of_date_au.normalize();
        let longitude = direction
            .y
            .atan2(direction.x)
            .to_degrees()
            .rem_euclid(360.0);
        let latitude = direction.z.asin().to_degrees();
        assert!(
            (longitude - 133.162_655).abs() < 1e-3,
            "lambda {longitude} deg"
        );
        assert!((latitude + 3.229_126).abs() < 1e-3, "beta {latitude} deg");
        assert!(
            (coordinates.distance_km - 368_409.7).abs() < 1.0,
            "Delta {} km",
            coordinates.distance_km
        );
    }

    /// The lunar distance must stay inside the published perigee/apogee range
    /// over a full anomalistic cycle.
    #[test]
    fn distance_stays_within_the_published_perigee_apogee_range() {
        let mut min = f64::INFINITY;
        let mut max: f64 = 0.0;
        for day in 0..400 {
            let km = geocentric_ecliptic(2_451_545.0 + f64::from(day))
                .unwrap()
                .distance_km;
            min = min.min(km);
            max = max.max(km);
        }
        assert!((356_000.0..362_000.0).contains(&min), "perigee {min} km");
        assert!((404_000.0..407_000.0).contains(&max), "apogee {max} km");
    }

    #[test]
    fn truncated_or_corrupt_assets_are_rejected() {
        assert_eq!(
            parse_bytes(&DATA[..DATA.len() - 1]).unwrap_err(),
            AstroError::InvalidData("moon_terms.bin")
        );
        let mut wrong_magic = DATA.to_vec();
        wrong_magic[3] = b'X';
        assert_eq!(
            parse_bytes(&wrong_magic).unwrap_err(),
            AstroError::InvalidData("moon_terms.bin")
        );
    }
}
