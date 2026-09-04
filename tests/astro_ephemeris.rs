use forge3d::astro::time::UtcDateTime;
use forge3d::astro::{body_position, moon_phase, Body, Observer};
use forge3d::geo::units::{Angle, Degree};

#[test]
fn all_sidera_bodies_produce_finite_topocentric_positions() {
    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap();
    let observer = Observer::new(Angle::<Degree>::new(52.3676), Angle::new(4.9041), 0.0).unwrap();
    for body in [
        Body::Sun,
        Body::Moon,
        Body::Mercury,
        Body::Venus,
        Body::Mars,
        Body::Jupiter,
        Body::Saturn,
    ] {
        let position = body_position(body, utc, observer, true).unwrap();
        assert!((0.0..360.0).contains(&position.azimuth.value()));
        assert!((-90.0..=90.0).contains(&position.altitude.value()));
        assert!(position.distance_au.is_finite() && position.distance_au > 0.0);
    }
}

#[test]
fn moon_phase_is_exactly_bounded() {
    let utc = UtcDateTime::new(2024, 4, 8, 18, 21, 0.0).unwrap();
    let phase = moon_phase(utc).unwrap();
    assert!((0.0..=1.0).contains(&phase.illuminated_fraction));
    assert!(phase.apparent_semidiameter_arcsec > 800.0);
    assert!(phase.apparent_semidiameter_arcsec < 1_100.0);
}
