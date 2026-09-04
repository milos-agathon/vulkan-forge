//! Bounded VSOP87D term evaluation over SIDERA's 2000–2050 window.
//!
//! The compact asset retains 14,793 of 25,659 IMCCE terms.  The generator
//! removes the smallest `|A*t^power|` contributions per body and coordinate,
//! subject to the aggregate bounds declared in `assets/astro/MANIFEST.toml`;
//! exact L/B/R counts by power are published there and locked by tests.

use super::AstroError;
use glam::DVec3;
use std::io::{Cursor, Read};
use std::sync::OnceLock;

const DATA: &[u8] = include_bytes!("../../assets/astro/vsop87d.bin");
const J2000: f64 = 2_451_545.0;

#[derive(Debug)]
struct Theory {
    bodies: Vec<BodySeries>,
    sections: Vec<Section>,
}

#[derive(Debug)]
struct BodySeries {
    name: [u8; 3],
    sections: Vec<usize>,
}

#[derive(Debug)]
struct Section {
    coordinate: usize,
    power: i32,
    terms: Vec<Term>,
}

#[derive(Debug)]
struct Term {
    amplitude: f64,
    phase: f64,
    frequency: f64,
}

#[derive(Clone, Copy)]
pub(crate) enum VsopBody {
    Mercury,
    Venus,
    Earth,
    Mars,
    Jupiter,
    Saturn,
}

pub(crate) fn heliocentric_ecliptic(body: VsopBody, jd_tt: f64) -> Result<DVec3, AstroError> {
    let name = match body {
        VsopBody::Mercury => *b"mer",
        VsopBody::Venus => *b"ven",
        VsopBody::Earth => *b"ear",
        VsopBody::Mars => *b"mar",
        VsopBody::Jupiter => *b"jup",
        VsopBody::Saturn => *b"sat",
    };
    let theory = theory()?;
    let body = theory
        .bodies
        .iter()
        .find(|body| body.name == name)
        .ok_or(AstroError::InvalidData("vsop87d.bin"))?;
    let t = (jd_tt - J2000) / 365_250.0;
    let mut lbr = [0.0; 3];
    for index in &body.sections {
        let section = theory
            .sections
            .get(*index)
            .ok_or(AstroError::InvalidData("vsop87d.bin"))?;
        let sum = section
            .terms
            .iter()
            .map(|term| term.amplitude * (term.phase + term.frequency * t).cos())
            .sum::<f64>();
        lbr[section.coordinate] += sum * t.powi(section.power);
    }
    let (sin_l, cos_l) = lbr[0].sin_cos();
    let (sin_b, cos_b) = lbr[1].sin_cos();
    Ok(DVec3::new(
        lbr[2] * cos_b * cos_l,
        lbr[2] * cos_b * sin_l,
        lbr[2] * sin_b,
    ))
}

pub fn earth_velocity(jd_tt: f64) -> Result<DVec3, AstroError> {
    const STEP_DAYS: f64 = 0.01;
    Ok((heliocentric_ecliptic(VsopBody::Earth, jd_tt + STEP_DAYS)?
        - heliocentric_ecliptic(VsopBody::Earth, jd_tt - STEP_DAYS)?)
        / (2.0 * STEP_DAYS))
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
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    if &magic != b"F3DVSOP1" {
        return Err(AstroError::InvalidData("vsop87d.bin"));
    }
    let body_count = read_u32(&mut input)? as usize;
    let section_count = read_u32(&mut input)? as usize;
    let mut bodies = Vec::with_capacity(body_count);
    for _ in 0..body_count {
        let mut name = [0; 3];
        input
            .read_exact(&mut name)
            .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
        let count = read_u8(&mut input)? as usize;
        let mut sections = Vec::with_capacity(count);
        for _ in 0..count {
            sections.push(read_u32(&mut input)? as usize);
        }
        bodies.push(BodySeries { name, sections });
    }
    let mut sections = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        let coordinate = read_u8(&mut input)? as usize;
        let power = read_u8(&mut input)? as i32;
        let _reserved = read_u16(&mut input)?;
        let count = read_u32(&mut input)? as usize;
        if coordinate > 2 {
            return Err(AstroError::InvalidData("vsop87d.bin"));
        }
        let mut terms = Vec::with_capacity(count);
        for _ in 0..count {
            terms.push(Term {
                amplitude: read_f64(&mut input)?,
                phase: read_f64(&mut input)?,
                frequency: read_f64(&mut input)?,
            });
        }
        sections.push(Section {
            coordinate,
            power,
            terms,
        });
    }
    if input.position() != data.len() as u64 {
        return Err(AstroError::InvalidData("vsop87d.bin"));
    }
    Ok(Theory { bodies, sections })
}

fn read_u8(input: &mut Cursor<&[u8]>) -> Result<u8, AstroError> {
    let mut bytes = [0; 1];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(bytes[0])
}

fn read_u16(input: &mut Cursor<&[u8]>) -> Result<u16, AstroError> {
    let mut bytes = [0; 2];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &mut Cursor<&[u8]>) -> Result<u32, AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f64(input: &mut Cursor<&[u8]>) -> Result<f64, AstroError> {
    let mut bytes = [0; 8];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(f64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const J2000_TT: f64 = 2_451_545.0;

    /// The committed asset must parse to the six declared bodies with every
    /// retained L/B/R power series, and consume every byte (the parser already
    /// rejects a trailing-byte mismatch, so reaching `Ok` proves exactness).
    #[test]
    fn committed_theory_round_trips_with_every_body_and_series() {
        let theory = theory().expect("committed VSOP87D asset parses");
        assert_eq!(theory.bodies.len(), 6);
        let mut names: Vec<&[u8]> = theory.bodies.iter().map(|body| &body.name[..]).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![&b"ear"[..], b"jup", b"mar", b"mer", b"sat", b"ven"]
        );
        let mut total_terms = 0usize;
        for body in &theory.bodies {
            let mut coordinates = [0usize; 3];
            for index in &body.sections {
                let section = &theory.sections[*index];
                coordinates[section.coordinate] += 1;
                total_terms += section.terms.len();
                assert!(!section.terms.is_empty());
                assert!((0..=5).contains(&section.power));
            }
            // VSOP87D publishes L0..L5, B0..B5, R0..R5 for every planet.
            assert!(
                coordinates.iter().all(|count| *count >= 5),
                "{} has thin series {coordinates:?}",
                String::from_utf8_lossy(&body.name)
            );
            let retained = body
                .sections
                .iter()
                .map(|index| theory.sections[*index].terms.len())
                .sum::<usize>();
            let expected = match &body.name {
                b"mer" => 2_620,
                b"ven" => 912,
                b"ear" => 1_348,
                b"mar" => 3_357,
                b"jup" => 2_421,
                b"sat" => 4_135,
                _ => unreachable!(),
            };
            assert_eq!(
                retained, expected,
                "unexpected truncation for {:?}",
                body.name
            );
        }
        assert_eq!(total_terms, 14_793);
        assert!(
            total_terms < 25_659,
            "the full source theory was not truncated"
        );
    }

    /// Earth's heliocentric distance oscillates between the published
    /// perihelion and aphelion; J2000.0 falls two days before perihelion.
    #[test]
    fn earth_radius_vector_brackets_the_published_orbit() {
        let r = heliocentric_ecliptic(VsopBody::Earth, J2000_TT)
            .unwrap()
            .length();
        assert!((r - 0.983_3).abs() < 1e-3, "J2000 Earth radius {r} AU");
        let mut min = f64::INFINITY;
        let mut max: f64 = 0.0;
        for day in 0..366 {
            let r = heliocentric_ecliptic(VsopBody::Earth, J2000_TT + f64::from(day))
                .unwrap()
                .length();
            min = min.min(r);
            max = max.max(r);
        }
        assert!((min - 0.983_29).abs() < 1e-3, "perihelion {min} AU");
        assert!((max - 1.016_71).abs() < 1e-3, "aphelion {max} AU");
    }

    /// Each body's radius vector must stay inside its published perihelion and
    /// aphelion over a full revolution — an independent check that each series
    /// is wired to the right coefficients. `(a, e)` from the JPL planetary
    /// fact sheets; the bracket is widened by 1% to absorb the fact that these
    /// are mean elements while VSOP87D returns osculating positions.
    #[test]
    fn planet_radius_vectors_stay_within_published_perihelion_and_aphelion() {
        let cases = [
            (VsopBody::Mercury, 0.387_1, 0.205_6, 88),
            (VsopBody::Venus, 0.723_3, 0.006_8, 225),
            (VsopBody::Earth, 1.000_0, 0.016_7, 365),
            (VsopBody::Mars, 1.523_7, 0.093_4, 687),
            (VsopBody::Jupiter, 5.203_0, 0.048_9, 4_333),
            (VsopBody::Saturn, 9.537_0, 0.056_5, 10_759),
        ];
        for (body, semi_major_axis, eccentricity, period_days) in cases {
            let mut min = f64::INFINITY;
            let mut max: f64 = 0.0;
            for step in 0..=period_days {
                let r = heliocentric_ecliptic(body, J2000_TT + f64::from(step))
                    .unwrap()
                    .length();
                min = min.min(r);
                max = max.max(r);
            }
            let perihelion = semi_major_axis * (1.0 - eccentricity);
            let aphelion = semi_major_axis * (1.0 + eccentricity);
            assert!(
                (min - perihelion).abs() < semi_major_axis * 0.01,
                "perihelion {min} AU vs published {perihelion} AU"
            );
            assert!(
                (max - aphelion).abs() < semi_major_axis * 0.01,
                "aphelion {max} AU vs published {aphelion} AU"
            );
        }
    }

    /// The finite-difference Earth velocity must match the published mean
    /// orbital speed of 29.78 km/s (0.0172 AU/day).
    #[test]
    fn earth_velocity_is_the_published_orbital_speed() {
        let speed = earth_velocity(J2000_TT).unwrap().length();
        assert!((speed - 0.017_2).abs() < 5e-4, "{speed} AU/day");
    }

    #[test]
    fn truncated_or_corrupt_assets_are_rejected() {
        assert_eq!(
            parse_bytes(&DATA[..DATA.len() - 1]).unwrap_err(),
            AstroError::InvalidData("vsop87d.bin")
        );
        let mut wrong_magic = DATA.to_vec();
        wrong_magic[0] = b'X';
        assert_eq!(
            parse_bytes(&wrong_magic).unwrap_err(),
            AstroError::InvalidData("vsop87d.bin")
        );
    }
}
