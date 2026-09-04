//! Reference-frame and observer reductions.

use super::Observer;
use crate::geo::units::{Angle, Degree};
use glam::{DMat3, DVec3};

const J2000: f64 = 2_451_545.0;
const DAYS_PER_CENTURY: f64 = 36_525.0;
const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3_600.0);
const AU_METRES: f64 = 149_597_870_700.0;
const LIGHT_SPEED_AU_PER_DAY: f64 = 173.144_632_684_669_3;

/// Greenwich mean sidereal angle, IAU 2006 (`iauGmst06`).
pub fn gmst(jd_ut1: f64, jd_tt: f64) -> f64 {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    normalize_radians(
        earth_rotation_angle(jd_ut1)
            + (0.014_506
                + (4_612.156_534
                    + (1.391_581_7
                        + (-0.000_000_44 + (-0.000_029_956 + -0.000_000_036_8 * t) * t) * t)
                        * t)
                    * t)
                * ARCSEC_TO_RAD,
    )
}

/// Greenwich apparent sidereal angle using the equation of the equinoxes.
///
/// Nutation is the four dominant IAU 1980/Meeus terms; their omitted
/// contribution stays below the SIDERA 0.1 sidereal-second budget in
/// 2000–2050 and is validated against the Horizons oracle.
pub fn gast(jd_ut1: f64, jd_tt: f64) -> f64 {
    let (dpsi, _, mean_obliquity) = nutation(jd_tt);
    normalize_radians(gmst(jd_ut1, jd_tt) + dpsi * mean_obliquity.cos())
}

/// IAU 2006 Fukushima–Williams precession, GCRS/J2000 to mean equator of date.
pub fn precess_j2000_to_date(direction: DVec3, jd_tt: f64) -> DVec3 {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    let gamb = polynomial(
        t,
        &[
            -0.052_928,
            10.556_378,
            0.493_204_4,
            -0.000_312_38,
            -0.000_002_788,
            0.000_000_026,
        ],
    ) * ARCSEC_TO_RAD;
    let phib = polynomial(
        t,
        &[
            84_381.412_819,
            -46.811_016,
            0.051_126_8,
            0.000_532_89,
            -0.000_000_44,
            -0.000_000_017_6,
        ],
    ) * ARCSEC_TO_RAD;
    let psib = polynomial(
        t,
        &[
            -0.041_775,
            5_038.481_484,
            1.558_417_5,
            -0.000_185_22,
            -0.000_026_452,
            -0.000_000_014_8,
        ],
    ) * ARCSEC_TO_RAD;
    let epsa = mean_obliquity(jd_tt);
    let matrix = DMat3::from_rotation_x(epsa)
        * DMat3::from_rotation_z(psib)
        * DMat3::from_rotation_x(-phib)
        * DMat3::from_rotation_z(-gamb);
    (matrix * direction).normalize()
}

/// Mean equator/equinox of date to true place using the dominant
/// IAU 1980/Meeus nutation terms declared by [`nutation`].
pub fn nutate_mean_to_true(direction: DVec3, jd_tt: f64) -> DVec3 {
    let (dpsi, deps, mean_obliquity) = nutation(jd_tt);
    (DMat3::from_rotation_x(mean_obliquity + deps)
        * DMat3::from_rotation_z(dpsi)
        * DMat3::from_rotation_x(-mean_obliquity)
        * direction)
        .normalize()
}

/// First-order annual aberration, equivalent to the vector form in SOFA
/// `iauAb` when gravitational potential and second-order beta terms are
/// omitted (<0.01 arcsec over SIDERA's window).
pub fn annual_aberration(direction: DVec3, earth_velocity_au_per_day: DVec3) -> DVec3 {
    let p = direction.normalize();
    let beta = earth_velocity_au_per_day / LIGHT_SPEED_AU_PER_DAY;
    (p + beta - p * p.dot(beta)).normalize()
}

/// WGS84 geodetic observer position in Earth-fixed equatorial axes, AU.
pub fn observer_ecef(observer: Observer) -> DVec3 {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    let e2 = F * (2.0 - F);
    let lat = observer.latitude.radians();
    let lon = observer.longitude.radians();
    let n = A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    DVec3::new(
        (n + observer.height_m) * lat.cos() * lon.cos(),
        (n + observer.height_m) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + observer.height_m) * lat.sin(),
    ) / AU_METRES
}

/// Geocentric true-equatorial vector to topocentric true-equatorial vector.
///
/// Subtracting the rotated WGS84 observer makes lunar horizontal parallax
/// (up to roughly one degree) explicit and independently ablatable.
pub fn geocentric_to_topocentric(
    geocentric_equatorial_au: DVec3,
    observer: Observer,
    apparent_sidereal_time: f64,
) -> DVec3 {
    let observer_inertial =
        DMat3::from_rotation_z(apparent_sidereal_time) * observer_ecef(observer);
    geocentric_equatorial_au - observer_inertial
}

pub fn equatorial_to_horizontal(
    topocentric_equatorial: DVec3,
    observer: Observer,
    apparent_sidereal_time: f64,
) -> (Angle<Degree>, Angle<Degree>) {
    let earth_fixed =
        DMat3::from_rotation_z(-apparent_sidereal_time) * topocentric_equatorial.normalize();
    let lat = observer.latitude.radians();
    let lon = observer.longitude.radians();
    let east = DVec3::new(-lon.sin(), lon.cos(), 0.0);
    let north = DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
    let up = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
    let e = earth_fixed.dot(east);
    let n = earth_fixed.dot(north);
    let u = earth_fixed.dot(up).clamp(-1.0, 1.0);
    let azimuth = normalize_radians(e.atan2(n)).to_degrees();
    (Angle::new(azimuth), Angle::new(u.asin().to_degrees()))
}

/// Sæmundsson (1986) true-altitude to apparent-altitude refraction.
///
/// `R = 1.02 / tan(h + 10.3/(h + 5.11))` arcminutes, as tabulated in Meeus,
/// *Astronomical Algorithms* 2nd ed., eq. 16.4. Declared standard atmosphere:
/// 1010 mbar at 10 °C. Outside `-1° <= h < 89°` the fit is not valid and the
/// altitude is returned unchanged rather than extrapolated.
pub fn refract_altitude(altitude: Angle<Degree>) -> Angle<Degree> {
    let degrees = altitude.value();
    if !degrees.is_finite() || !(-1.0..89.0).contains(&degrees) {
        return altitude;
    }
    let correction_arcminutes = 1.02 / (degrees + 10.3 / (degrees + 5.11)).to_radians().tan();
    Angle::new(degrees + correction_arcminutes / 60.0)
}

/// Bennett (1982) refraction in arcminutes for an *apparent* altitude.
///
/// `R = 1 / tan(h + 7.31/(h + 4.4))`, Meeus eq. 16.3, same declared standard
/// atmosphere. This is an independently published fit in the opposite
/// direction from [`refract_altitude`]; SIDERA's validation metrics gate the
/// two against each other rather than against a copy of either one's output.
pub fn bennett_refraction_arcminutes(apparent_altitude: Angle<Degree>) -> f64 {
    let degrees = apparent_altitude.value();
    1.0 / (degrees + 7.31 / (degrees + 4.4)).to_radians().tan()
}

/// True altitude whose [`refract_altitude`] image is `apparent`, or `None`.
///
/// Fixed-point inversion of the Sæmundsson fit, seeded from the Bennett value;
/// `dR/dh` is far below 1 inside the domain, so the iteration contracts.
///
/// Returns `None` outside the *image* of [`refract_altitude`] — below roughly
/// −0.353° apparent there is no true altitude that refracts to the requested
/// value, and above 89° the fit does not apply. Returning `None` rather than
/// the input is deliberate: an unconverged iterate that happens to echo its
/// argument would read as "no refraction" exactly in the near-horizon regime
/// where refraction matters most.
pub fn unrefract_altitude(apparent: Angle<Degree>) -> Option<Angle<Degree>> {
    const CONVERGENCE_DEG: f64 = 1e-12;
    let target = apparent.value();
    let domain_floor = refract_altitude(Angle::<Degree>::new(-1.0)).value();
    if !target.is_finite() || !(domain_floor..89.0).contains(&target) {
        return None;
    }
    let mut true_altitude = target - bennett_refraction_arcminutes(apparent) / 60.0;
    for _ in 0..64 {
        let error = refract_altitude(Angle::new(true_altitude)).value() - target;
        if error.abs() < CONVERGENCE_DEG {
            return Some(Angle::new(true_altitude));
        }
        true_altitude -= error;
    }
    None
}

/// Greenwich mean sidereal time in seconds, IAU 1982 (Aoki et al.).
///
/// `GMST = 67310.54841 + 3164400184.812866 T + 0.093104 T² - 6.2e-6 T³`
/// seconds, `T` in Julian centuries of UT1 from J2000; USNO Circular 179
/// eq. 2.24 / Meeus eq. 12.4. This is an algebraically independent route to
/// sidereal time from the ERA-based [`gmst`] above (which is IAU 2006 and also
/// consumes TT), and exists so the two can be cross-gated.
pub fn gmst_iau1982_seconds(jd_ut1: f64) -> f64 {
    let t = (jd_ut1 - J2000) / DAYS_PER_CENTURY;
    polynomial(
        t,
        &[67_310.548_41, 3_164_400_184.812_866, 0.093_104, -6.2e-6],
    )
    .rem_euclid(86_400.0)
}

pub fn mean_obliquity(jd_tt: f64) -> f64 {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    polynomial(
        t,
        &[
            84_381.406,
            -46.836_769,
            -0.000_183_1,
            0.002_003_4,
            -0.000_000_576,
            -0.000_000_043_4,
        ],
    ) * ARCSEC_TO_RAD
}

pub fn nutation(jd_tt: f64) -> (f64, f64, f64) {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    let l = (280.4665 + 36_000.769_8 * t).to_radians();
    let lp = (218.3165 + 481_267.881_3 * t).to_radians();
    let omega = (125.04452 - 1_934.136_261 * t).to_radians();
    let dpsi_arcsec = -17.20 * omega.sin() - 1.32 * (2.0 * l).sin() - 0.23 * (2.0 * lp).sin()
        + 0.21 * (2.0 * omega).sin();
    let deps_arcsec = 9.20 * omega.cos() + 0.57 * (2.0 * l).cos() + 0.10 * (2.0 * lp).cos()
        - 0.09 * (2.0 * omega).cos();
    (
        dpsi_arcsec * ARCSEC_TO_RAD,
        deps_arcsec * ARCSEC_TO_RAD,
        mean_obliquity(jd_tt),
    )
}

fn earth_rotation_angle(jd_ut1: f64) -> f64 {
    let d = jd_ut1 - J2000;
    let fraction = jd_ut1.rem_euclid(1.0);
    normalize_radians(
        std::f64::consts::TAU * (fraction + 0.779_057_273_264 + 0.002_737_811_911_354_48 * d),
    )
}

fn normalize_radians(value: f64) -> f64 {
    value.rem_euclid(std::f64::consts::TAU)
}

fn polynomial(t: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * t + coefficient)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Meeus, *Astronomical Algorithms* 2nd ed., examples 12.a and 12.b.
    /// Both epochs sit outside SIDERA's 2000–2050 public window on purpose:
    /// the sidereal-time kernel itself takes raw Julian dates, so the textbook
    /// identities can be checked where the textbook states them.
    #[test]
    fn gmst_matches_meeus_textbook_epochs() {
        // 1987 April 10, 0h UT -> 13h 10m 46.3668s.
        let a = gmst_iau1982_seconds(2_446_895.5);
        assert!(
            (a - (13.0 * 3_600.0 + 10.0 * 60.0 + 46.366_8)).abs() < 1e-3,
            "example 12.a gave {a} s"
        );
        // 1987 April 10, 19h 21m 00s UT -> 8h 34m 57.0896s.
        let b = gmst_iau1982_seconds(2_446_896.306_25);
        assert!(
            (b - (8.0 * 3_600.0 + 34.0 * 60.0 + 57.089_6)).abs() < 1e-3,
            "example 12.b gave {b} s"
        );
    }

    /// The IAU 2006 ERA-based kernel and the IAU 1982 polynomial are separate
    /// algorithms; they must still agree far inside the 0.1 s SIDERA budget.
    #[test]
    fn iau2006_and_iau1982_sidereal_time_agree_across_the_window() {
        let mut worst: f64 = 0.0;
        // 2000-01-01T12:00 UT1 through 2050-12-31, sampled every 37 days.
        let mut jd = J2000;
        while jd < J2000 + 18_628.0 {
            let modern = gmst(jd, jd) * 86_400.0 / std::f64::consts::TAU;
            let classical = gmst_iau1982_seconds(jd);
            let mut delta = modern - classical;
            // Both are wrapped to a day; compare on the circle.
            delta -= (delta / 86_400.0).round() * 86_400.0;
            worst = worst.max(delta.abs());
            jd += 37.0;
        }
        assert!(worst < 0.1, "worst GMST disagreement {worst} s");
    }

    /// Sæmundsson (true -> apparent) inverted at 5° apparent altitude must
    /// reproduce Bennett (apparent -> refraction) to well inside 0.2'.
    #[test]
    fn saemundsson_and_bennett_agree_at_five_degrees_apparent() {
        let apparent = Angle::<Degree>::new(5.0);
        let sidera = (5.0 - unrefract_altitude(apparent).unwrap().value()) * 60.0;
        let bennett = bennett_refraction_arcminutes(apparent);
        assert!(
            sidera > 9.0 && sidera < 11.0,
            "implausible refraction {sidera}'"
        );
        assert!(
            (sidera - bennett).abs() < 0.2,
            "sidera {sidera}' vs bennett {bennett}'"
        );
    }

    #[test]
    fn unrefract_inverts_refract_and_refuses_to_extrapolate() {
        for true_altitude in [-1.0_f64, -0.5, 0.0, 2.5, 5.0, 20.0, 60.0, 85.0, 88.9] {
            let apparent = refract_altitude(Angle::new(true_altitude));
            let recovered = unrefract_altitude(apparent)
                .unwrap_or_else(|| panic!("{true_altitude}° is inside the fit's image"))
                .value();
            assert!(
                (recovered - true_altitude).abs() < 1e-9,
                "{true_altitude}° round-tripped to {recovered}°"
            );
        }
        // Below the image of the fit there is no answer, and the old fixed
        // point silently echoed its argument here (reading as zero refraction).
        let floor = refract_altitude(Angle::<Degree>::new(-1.0)).value();
        assert!(floor > -0.4 && floor < -0.3, "image floor {floor}°");
        for outside in [floor - 1e-6, -0.4, -1.0, -5.0, 89.0, 95.0, f64::NAN] {
            assert!(
                unrefract_altitude(Angle::new(outside)).is_none(),
                "{outside}° should be outside the invertible domain"
            );
        }
    }

    /// Mean obliquity at J2000 is 23° 26' 21.406" by definition of the IAU 2006
    /// series (Hilton et al. 2006), and it decreases with time.
    #[test]
    fn mean_obliquity_matches_iau2006_at_j2000() {
        let epsilon = mean_obliquity(J2000).to_degrees() * 3_600.0;
        assert!((epsilon - 84_381.406).abs() < 1e-6, "{epsilon}\"");
        assert!(mean_obliquity(J2000 + 36_525.0) < mean_obliquity(J2000));
    }

    /// General precession in longitude is about 50.29"/yr, so an equatorial
    /// direction must move by roughly 26 arcminutes over the 2000–2026 span.
    #[test]
    fn precession_rate_is_of_the_published_magnitude() {
        let arcsec = DVec3::X
            .angle_between(precess_j2000_to_date(DVec3::X, J2000 + 36_525.0))
            .to_degrees()
            * 3_600.0;
        // One century of general precession on the equinox: ~5029".
        assert!(
            (4_800.0..5_300.0).contains(&arcsec),
            "{arcsec}\" per century"
        );
    }
}
