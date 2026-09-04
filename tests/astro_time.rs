use forge3d::astro::frames::{annual_aberration, gmst, precess_j2000_to_date, refract_altitude};
use forge3d::astro::time::{
    julian_day_tt, julian_day_ut1, julian_day_utc, tai_minus_utc_seconds, UtcDateTime,
};
use forge3d::geo::units::{Angle, Degree};
use glam::DVec3;

#[test]
fn j2000_utc_identity_and_window_are_enforced() {
    let epoch = UtcDateTime::new(2000, 1, 1, 12, 0, 0.0).unwrap();
    assert!((julian_day_utc(epoch) - 2_451_545.0).abs() < 1e-9);
    assert!((tai_minus_utc_seconds(epoch).unwrap() - 32.0).abs() < f64::EPSILON);
    assert!((julian_day_tt(epoch).unwrap() - (2_451_545.0 + 64.184 / 86_400.0)).abs() < 1e-12);
    assert!((julian_day_ut1(epoch).unwrap() - julian_day_utc(epoch)).abs() > 1e-9);
    assert!(UtcDateTime::new(1999, 12, 31, 23, 59, 59.0).is_err());
    assert!(UtcDateTime::new(2051, 1, 1, 0, 0, 0.0).is_err());
}

#[test]
fn utc_accepts_only_real_positive_leap_seconds() {
    let leap = UtcDateTime::new(2016, 12, 31, 23, 59, 60.0).unwrap();
    let after = UtcDateTime::new(2017, 1, 1, 0, 0, 0.0).unwrap();
    assert!(julian_day_utc(leap) < julian_day_utc(after));
    assert!(
        ((julian_day_tt(after).unwrap() - julian_day_tt(leap).unwrap()) * 86_400.0 - 1.0).abs()
            < 1e-5
    );
    assert!(UtcDateTime::new(2016, 12, 30, 23, 59, 60.0).is_err());
    assert!(UtcDateTime::new(2016, 12, 31, 22, 59, 60.0).is_err());
}

#[test]
fn iau_2006_gmst_matches_sofa_reference_epoch() {
    // IAU SOFA t_sofa_c.c reference for iauGmst06 at MJD 53736.0.
    let jd = 2_400_000.5 + 53_736.0;
    let actual = gmst(jd, jd);
    let expected = 1.754_174_971_870_091_2;
    let error_seconds = (actual - expected).abs() * 86_400.0 / (2.0 * std::f64::consts::PI);
    assert!(error_seconds < 0.1, "{error_seconds} sidereal seconds");
}

#[test]
fn iau_2006_gmst_stays_within_budget_across_sidera_window() {
    let cases = [
        (
            UtcDateTime::new(2000, 1, 1, 0, 0, 0.0).unwrap(),
            1.744_793_167_080_565_2,
        ),
        (
            UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap(),
            4.792_809_523_685_919,
        ),
        (
            UtcDateTime::new(2050, 12, 31, 23, 0, 0.0).unwrap(),
            1.493_405_560_994_384_4,
        ),
    ];
    for (utc, expected) in cases {
        let actual = gmst(julian_day_ut1(utc).unwrap(), julian_day_tt(utc).unwrap());
        let error_seconds = (actual - expected).abs() * 86_400.0 / (2.0 * std::f64::consts::PI);
        assert!(error_seconds < 0.1, "{error_seconds} sidereal seconds");
    }
}

#[test]
fn bennett_refraction_at_five_degrees_matches_reference() {
    let apparent = refract_altitude(Angle::<Degree>::new(5.0));
    let correction_arcminutes = (apparent.value() - 5.0) * 60.0;
    assert!((correction_arcminutes - 9.674_126_604).abs() < 0.2);
}

#[test]
fn frame_reductions_change_direction_and_preserve_unit_length() {
    let epoch = UtcDateTime::new(2026, 7, 26, 0, 0, 0.0).unwrap();
    let jd_tt = julian_day_tt(epoch).unwrap();
    let j2000 = DVec3::new(0.3, -0.4, 0.866_025_403_784).normalize();
    let precessed = precess_j2000_to_date(j2000, jd_tt);
    assert!((precessed.length() - 1.0).abs() < 1e-12);
    assert!(precessed.angle_between(j2000).to_degrees() > 0.1);

    let aberrated = annual_aberration(precessed, DVec3::new(0.0, 0.017_2, 0.0));
    assert!((aberrated.length() - 1.0).abs() < 1e-12);
    assert!(aberrated.angle_between(precessed) > 0.0);
}

#[test]
fn iau_2006_precession_matches_sofa_reference_vector() {
    // SOFA iauPmat06 at JD(TT) 2460680.5 applied to this unit vector.
    let input = DVec3::new(0.3, -0.4, 0.866_025_403_784).normalize();
    let expected = DVec3::new(
        0.300_127_195_473_648_63,
        -0.398_321_626_127_662_6,
        0.866_754_606_965_626_1,
    );
    let actual = precess_j2000_to_date(input, 2_460_680.5);
    assert!(
        actual.angle_between(expected).to_degrees() * 3_600.0 < 1e-3,
        "{actual:?}"
    );
}
