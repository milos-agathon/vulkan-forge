use forge3d::astro::{
    catalog::{
        bright_star_catalog, bv_to_linear_rgb, magnitude_to_irradiance, star_instances,
        twilight_visibility,
    },
    frames,
    time::{julian_day_tt, julian_day_ut1, UtcDateTime},
    vsop, Observer,
};
use forge3d::geo::units::Angle;
use glam::{DMat3, DVec3};

#[test]
fn committed_bright_star_catalog_is_complete_and_bounded() {
    let stars = bright_star_catalog().expect("catalog");
    assert!((9_000..=9_110).contains(&stars.len()), "{}", stars.len());
    assert!(stars.iter().all(|star| {
        star.ra_j2000().value().is_finite()
            && (0.0..360.0).contains(&star.ra_j2000().value())
            && (-90.0..=90.0).contains(&star.dec_j2000().value())
            && star.v_magnitude().is_finite()
    }));
}

#[test]
fn catalog_photometry_and_per_frame_transform_are_finite() {
    let five_magnitudes = magnitude_to_irradiance(0.0) / magnitude_to_irradiance(5.0);
    assert!((five_magnitudes - 100.0).abs() < 1.0e-12);
    for bv in [-0.4, 0.0, 0.65, 1.5, 2.0] {
        assert!(bv_to_linear_rgb(bv)
            .iter()
            .all(|component| component.is_finite() && (0.0..=1.0).contains(component)));
    }
    assert_eq!(twilight_visibility(Angle::new(-4.0)), 0.0);
    assert_eq!(twilight_visibility(Angle::new(-18.0)), 1.0);

    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap();
    let observer = Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0).unwrap();
    let instances = star_instances(utc, observer, Angle::new(-18.0)).expect("instances");
    assert_eq!(instances.len(), bright_star_catalog().unwrap().len());
    assert!(instances.iter().all(|star| {
        star.azimuth.value().is_finite()
            && star.altitude.value().is_finite()
            && star.irradiance_w_m2.is_finite()
            && star
                .linear_rgb
                .iter()
                .all(|component| component.is_finite())
    }));

    // The physical irradiance must be exactly the cited photometric value with
    // NO rendering ramp baked in; the ramp lives in its own field. Daylight
    // must therefore zero `visibility` while leaving `irradiance_w_m2` intact.
    let daylight = star_instances(utc, observer, Angle::new(30.0)).expect("instances");
    for (night, day) in instances.iter().zip(&daylight) {
        assert_eq!(night.visibility, 1.0);
        assert_eq!(day.visibility, 0.0);
        assert_eq!(night.irradiance_w_m2, day.irradiance_w_m2);
        assert_eq!(
            night.irradiance_w_m2,
            magnitude_to_irradiance(f64::from(night.v_magnitude))
        );
    }
}

#[test]
fn catalog_instances_are_apparent_not_mean_place() {
    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap();
    let observer = Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0).unwrap();
    let star = bright_star_catalog().unwrap()[0];
    let ra = star.ra_j2000().radians();
    let dec = star.dec_j2000().radians();
    let j2000 = DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
    let jd_tt = julian_day_tt(utc).unwrap();
    let mean = frames::precess_j2000_to_date(j2000, jd_tt);
    let true_place = frames::nutate_mean_to_true(mean, jd_tt);
    let velocity = DMat3::from_rotation_x(frames::mean_obliquity(jd_tt))
        * vsop::earth_velocity(jd_tt).unwrap();
    let apparent = frames::annual_aberration(true_place, velocity);
    let expected = frames::equatorial_to_horizontal(
        apparent,
        observer,
        frames::gast(julian_day_ut1(utc).unwrap(), jd_tt),
    );
    let actual = star_instances(utc, observer, Angle::new(-18.0)).unwrap()[0];
    assert!((actual.azimuth.value() - expected.0.value()).abs() < 1.0e-10);
    assert!((actual.altitude.value() - expected.1.value()).abs() < 1.0e-10);

    // Agreeing with a mirror of the same pipeline proves nothing on its own:
    // both could skip every reduction. Assert the reductions actually move the
    // star, so a stubbed `star_instances` that returned the mean place would
    // fail here even though the mirror above would still agree with it.
    let mean_place = frames::equatorial_to_horizontal(
        j2000,
        observer,
        frames::gast(julian_day_ut1(utc).unwrap(), jd_tt),
    );
    let displacement_arcmin = separation_arcmin(
        actual.azimuth.value(),
        actual.altitude.value(),
        mean_place.0.value(),
        mean_place.1.value(),
    );
    assert!(
        displacement_arcmin > 15.0,
        "J2000 -> apparent moved only {displacement_arcmin}'"
    );
}

fn separation_arcmin(az_a: f64, alt_a: f64, az_b: f64, alt_b: f64) -> f64 {
    let delta_azimuth = (az_a - az_b).to_radians();
    let (alt_a, alt_b) = (alt_a.to_radians(), alt_b.to_radians());
    (alt_a.sin() * alt_b.sin() + alt_a.cos() * alt_b.cos() * delta_azimuth.cos())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
        * 60.0
}
