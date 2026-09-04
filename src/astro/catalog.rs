//! Yale Bright Star Catalogue rendering data (FK5/J2000, epoch 2000.0).
//!
//! Proper motion is intentionally outside SIDERA's 2000–2050 contract. The
//! per-frame transform applies IAU 2006 precession, nutation, and annual
//! aberration before converting to local horizontal coordinates. Star
//! visibility uses the declared **SIDERA
//! civil-to-astronomical smoothstep** rendering ramp: zero at solar altitude
//! −4° and full at −18°. It is not a physical sky-luminance claim.

use super::{frames, time, AstroError, Observer};
use crate::geo::units::{Angle, Degree};
use glam::DVec3;
use std::sync::OnceLock;

const DATA: &[u8] = include_bytes!("../../assets/astro/bright_stars.bin");
const HEADER_BYTES: usize = 12;
const RECORD_BYTES: usize = 16;
/// Band-integrated Johnson V irradiance of a V = 0 star above the atmosphere.
///
/// Derived, not guessed: Allen's *Astrophysical Quantities* (Cox, 4th ed.,
/// table 15.7) tabulates the V zero point as a **spectral flux density**
/// `f_λ(0) = 3.63e-9 erg cm⁻² s⁻¹ Å⁻¹ = 3.63e-12 W m⁻² Å⁻¹`, over an effective
/// bandwidth `Δλ ≈ 880 Å` centred at 0.545 µm. Integrating gives
/// `3.63e-12 × 880 ≈ 3.19e-9 W m⁻²`.
///
/// The per-Ångström figure is nearly 11× larger than the integrated one, and
/// quoting it as an irradiance would put the Sun's V-band irradiance above the
/// whole solar constant — see `sun_v_band_irradiance_is_below_the_solar_constant`.
const V_ZERO_IRRADIANCE_W_M2: f64 = 3.19e-9;
/// Substituted where the BSC5 leaves B−V blank; 0.65 is the solar value.
const DEFAULT_B_V: f32 = 0.65;

#[derive(Clone, Copy, Debug)]
pub struct CatalogStar {
    ra_j2000_deg: f32,
    dec_j2000_deg: f32,
    v_magnitude: f32,
    b_v: f32,
}

impl CatalogStar {
    pub fn ra_j2000(self) -> Angle<Degree> {
        Angle::new(self.ra_j2000_deg as f64)
    }

    pub fn dec_j2000(self) -> Angle<Degree> {
        Angle::new(self.dec_j2000_deg as f64)
    }

    pub fn v_magnitude(self) -> f32 {
        self.v_magnitude
    }

    pub fn b_v(self) -> Option<f32> {
        self.b_v.is_finite().then_some(self.b_v)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StarInstance {
    pub azimuth: Angle<Degree>,
    pub altitude: Angle<Degree>,
    /// Johnson V-band irradiance above the atmosphere, W/m², from the cited
    /// zero point. This is a photometric quantity: no rendering ramp is baked
    /// into it. The declared twilight ramp is kept separately in
    /// [`StarInstance::visibility`] so the two can never be confused.
    pub irradiance_w_m2: f64,
    /// Declared SIDERA twilight ramp in `[0, 1]` — a *rendering* model keyed to
    /// solar depression, not a sky-luminance claim. See [`twilight_visibility`].
    pub visibility: f64,
    pub linear_rgb: [f32; 3],
    pub v_magnitude: f32,
}

pub fn bright_star_catalog() -> Result<&'static [CatalogStar], AstroError> {
    static CATALOG: OnceLock<Result<Vec<CatalogStar>, AstroError>> = OnceLock::new();
    CATALOG
        .get_or_init(|| parse_catalog(DATA))
        .as_deref()
        .map_err(Clone::clone)
}

pub fn star_instances(
    utc: time::UtcDateTime,
    observer: Observer,
    solar_altitude: Angle<Degree>,
) -> Result<Vec<StarInstance>, AstroError> {
    let jd_tt = time::julian_day_tt(utc)?;
    let sidereal = frames::gast(time::julian_day_ut1(utc)?, jd_tt);
    let velocity_equatorial = glam::DMat3::from_rotation_x(frames::mean_obliquity(jd_tt))
        * super::vsop::earth_velocity(jd_tt)?;
    let visibility = twilight_visibility(solar_altitude);
    Ok(bright_star_catalog()?
        .iter()
        .map(|star| {
            let ra = star.ra_j2000().radians();
            let dec = star.dec_j2000().radians();
            let j2000 = DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
            let mean = frames::precess_j2000_to_date(j2000, jd_tt);
            let true_place = frames::nutate_mean_to_true(mean, jd_tt);
            let apparent = frames::annual_aberration(true_place, velocity_equatorial);
            let (azimuth, altitude) =
                frames::equatorial_to_horizontal(apparent, observer, sidereal);
            StarInstance {
                azimuth,
                altitude,
                irradiance_w_m2: magnitude_to_irradiance(star.v_magnitude as f64),
                visibility,
                // The BSC5 leaves B-V blank for a few hundred entries; those
                // are drawn at the solar-analogue colour rather than dropped.
                linear_rgb: bv_to_linear_rgb(star.b_v().unwrap_or(DEFAULT_B_V)),
                v_magnitude: star.v_magnitude,
            }
        })
        .collect())
}

/// Star displacement, in arcminutes, caused by removing precession alone.
///
/// DoD gate 5 is stated about *star positions*, so it is measured here on the
/// real rendered set: every catalog star is taken through
/// [`star_instances`]'s exact reduction chain twice, once with
/// [`frames::precess_j2000_to_date`] and once without, and the horizontal
/// separation of the two is reported. Returns `(min, median, max)`.
///
/// Precession is a rotation about the ecliptic pole, so the displacement scales
/// with a star's angular distance from that pole: the minimum is near zero by
/// geometry and the median is the honest summary of the rendered sky.
pub fn precession_ablation_arcminutes(
    utc: time::UtcDateTime,
    observer: Observer,
) -> Result<(f64, f64, f64), AstroError> {
    let jd_tt = time::julian_day_tt(utc)?;
    let sidereal = frames::gast(time::julian_day_ut1(utc)?, jd_tt);
    let velocity_equatorial = glam::DMat3::from_rotation_x(frames::mean_obliquity(jd_tt))
        * super::vsop::earth_velocity(jd_tt)?;
    let mut displacements: Vec<f64> = bright_star_catalog()?
        .iter()
        .map(|star| {
            let ra = star.ra_j2000().radians();
            let dec = star.dec_j2000().radians();
            let j2000 = DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
            let horizontal = |mean: DVec3| {
                let true_place = frames::nutate_mean_to_true(mean, jd_tt);
                let apparent = frames::annual_aberration(true_place, velocity_equatorial);
                frames::equatorial_to_horizontal(apparent, observer, sidereal)
            };
            let with = horizontal(frames::precess_j2000_to_date(j2000, jd_tt));
            let without = horizontal(j2000);
            separation_arcminutes(with, without)
        })
        .collect();
    if displacements.is_empty() {
        return Err(AstroError::InvalidData("bright_stars.bin"));
    }
    displacements.sort_by(f64::total_cmp);
    Ok((
        displacements[0],
        displacements[displacements.len() / 2],
        displacements[displacements.len() - 1],
    ))
}

fn separation_arcminutes(
    a: (Angle<Degree>, Angle<Degree>),
    b: (Angle<Degree>, Angle<Degree>),
) -> f64 {
    let delta_azimuth = (a.0.value() - b.0.value()).to_radians();
    let (alt_a, alt_b) = (a.1.radians(), b.1.radians());
    (alt_a.sin() * alt_b.sin() + alt_a.cos() * alt_b.cos() * delta_azimuth.cos())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
        * 60.0
}

/// Band-integrated Johnson V irradiance in W/m² for a given V magnitude, from
/// the zero point derived at [`V_ZERO_IRRADIANCE_W_M2`].
pub fn magnitude_to_irradiance(v_magnitude: f64) -> f64 {
    V_ZERO_IRRADIANCE_W_M2 * 10.0_f64.powf(-0.4 * v_magnitude)
}

/// B−V to temperature via Ballesteros (2012, EPL 97 34008), followed by a
/// compact blackbody-to-sRGB approximation for display color.
pub fn bv_to_linear_rgb(b_v: f32) -> [f32; 3] {
    let bv = b_v.clamp(-0.4, 2.0) as f64;
    let kelvin = 4_600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    let temperature = kelvin / 100.0;
    let (red, green, blue) = if temperature <= 66.0 {
        (
            255.0,
            (99.470_802_586_1 * temperature.ln() - 161.119_568_166_1).clamp(0.0, 255.0),
            if temperature <= 19.0 {
                0.0
            } else {
                (138.517_731_223_1 * (temperature - 10.0).ln() - 305.044_792_730_7)
                    .clamp(0.0, 255.0)
            },
        )
    } else {
        (
            (329.698_727_446 * (temperature - 60.0).powf(-0.133_204_759_2)).clamp(0.0, 255.0),
            (288.122_169_528_3 * (temperature - 60.0).powf(-0.075_514_849_2)).clamp(0.0, 255.0),
            255.0,
        )
    };
    [red, green, blue].map(|value| srgb_to_linear((value / 255.0) as f32))
}

pub fn twilight_visibility(solar_altitude: Angle<Degree>) -> f64 {
    let t = ((-solar_altitude.value() - 4.0) / 14.0).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn parse_catalog(data: &[u8]) -> Result<Vec<CatalogStar>, AstroError> {
    let invalid = || AstroError::InvalidData("bright_stars.bin");
    if data.len() < HEADER_BYTES || &data[..8] != b"F3DBSC01" {
        return Err(invalid());
    }
    let count = u32::from_le_bytes(data[8..12].try_into().map_err(|_| invalid())?) as usize;
    let expected = count
        .checked_mul(RECORD_BYTES)
        .and_then(|bytes| HEADER_BYTES.checked_add(bytes))
        .ok_or_else(invalid)?;
    if data.len() != expected {
        return Err(invalid());
    }
    data[HEADER_BYTES..]
        .chunks_exact(RECORD_BYTES)
        .map(|record| {
            let value = |start| {
                f32::from_le_bytes(
                    record[start..start + 4]
                        .try_into()
                        .expect("four-byte catalog field"),
                )
            };
            let star = CatalogStar {
                ra_j2000_deg: value(0),
                dec_j2000_deg: value(4),
                v_magnitude: value(8),
                b_v: value(12),
            };
            if !star.ra_j2000_deg.is_finite()
                || !(0.0..360.0).contains(&star.ra_j2000_deg)
                || !star.dec_j2000_deg.is_finite()
                || !(-90.0..=90.0).contains(&star.dec_j2000_deg)
                || !star.v_magnitude.is_finite()
                || star.b_v.is_infinite()
            {
                return Err(invalid());
            }
            Ok(star)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_record_round_trip() {
        let values = [101.25_f32, -16.7, -1.46, 0.0];
        let mut data = b"F3DBSC01\x01\0\0\0".to_vec();
        data.extend(values.into_iter().flat_map(f32::to_le_bytes));
        let parsed = parse_catalog(&data).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].ra_j2000_deg, values[0]);
        assert_eq!(parsed[0].dec_j2000_deg, values[1]);
        assert_eq!(parsed[0].v_magnitude, values[2]);
        assert_eq!(parsed[0].b_v, values[3]);

        data[24..28].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert_eq!(
            parse_catalog(&data).unwrap_err(),
            AstroError::InvalidData("bright_stars.bin")
        );
    }

    /// Anchors the photometric zero point's *absolute* scale, which the
    /// magnitude-ratio tests cannot see because the constant cancels in a
    /// ratio. The Sun is V = −26.74; its V-band irradiance must be a sensible
    /// fraction of the 1361 W/m² total solar irradiance, and certainly not
    /// larger than it. The per-Ångström Allen figure fails this by ~11×.
    #[test]
    fn sun_v_band_irradiance_is_below_the_solar_constant() {
        const SOLAR_CONSTANT_W_M2: f64 = 1_361.0;
        let sun = magnitude_to_irradiance(-26.74);
        assert!(
            sun < SOLAR_CONSTANT_W_M2,
            "V-band alone would be {sun} W/m^2, above the whole solar constant"
        );
        // The V passband is ~88 nm of a spectrum peaking near 500 nm, so a
        // low-double-digit percentage of the total is the expected order.
        let fraction = sun / SOLAR_CONSTANT_W_M2;
        assert!(
            (0.05..0.25).contains(&fraction),
            "V-band fraction of the solar constant is {fraction}"
        );
    }

    /// Magnitudes are logarithmic: five of them are exactly a factor 100.
    #[test]
    fn five_magnitudes_are_a_factor_of_one_hundred() {
        let ratio = magnitude_to_irradiance(0.0) / magnitude_to_irradiance(5.0);
        assert!((ratio - 100.0).abs() < 1e-12, "{ratio}");
    }
}
