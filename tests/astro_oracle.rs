use forge3d::astro::frames;
use forge3d::astro::moon;
use forge3d::astro::time::{delta_t_seconds, julian_day_tt, julian_day_ut1, UtcDateTime};
use forge3d::astro::{body_position, moon_phase, Body, Observer};
use forge3d::geo::units::{Angle, Degree};
use forge3d::lighting::ephemeris::sun_position;
use glam::DMat3;

#[test]
fn positions_meet_committed_horizons_thresholds() {
    let mut maxima = [0.0_f64; 7];
    let mut noaa_sun_max = 0.0_f64;
    let mut semidiameter_max = 0.0_f64;
    for line in include_str!("data/horizons_vectors.dat").lines() {
        if line.is_empty() || line.starts_with('#') || line.starts_with('@') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        let body = Body::parse(fields[2]).unwrap();
        let index = body_index(body);
        let utc = parse_utc(fields[1]);
        let observer = Observer::new(
            Angle::<Degree>::new(fields[3].parse().unwrap()),
            Angle::new(fields[4].parse().unwrap()),
            fields[5].parse().unwrap(),
        )
        .unwrap();
        let actual = body_position(body, utc, observer, false).unwrap();
        let expected_azimuth: f64 = fields[6].parse().unwrap();
        let expected_altitude: f64 = fields[7].parse().unwrap();
        maxima[index] = maxima[index].max(separation_arcsec(
            actual.azimuth.value(),
            actual.altitude.value(),
            expected_azimuth,
            expected_altitude,
        ));
        if body == Body::Sun {
            let (year, month, day, hour, minute, second) = utc.components();
            let baseline = sun_position(
                observer.latitude().value(),
                observer.longitude().value(),
                year,
                u32::from(month),
                u32::from(day),
                u32::from(hour),
                u32::from(minute),
                second as u32,
            );
            noaa_sun_max = noaa_sun_max.max(separation_arcsec(
                baseline.azimuth,
                baseline.elevation,
                expected_azimuth,
                expected_altitude,
            ));
        }
        if body == Body::Moon {
            let expected_radius = fields[9].parse::<f64>().unwrap() * 0.5;
            let actual_radius = (1_737.4 / (actual.distance_au * 149_597_870.7))
                .asin()
                .to_degrees()
                * 3_600.0;
            semidiameter_max = semidiameter_max.max((actual_radius - expected_radius).abs());
        }
    }
    println!("max deviations arcsec: {maxima:?}; NOAA Sun baseline: {noaa_sun_max}");
    assert!(maxima[0] <= 10.0, "Sun: {} arcsec", maxima[0]);
    assert!(maxima[0] < noaa_sun_max);
    assert!(maxima[1] <= 30.0, "Moon: {} arcsec", maxima[1]);
    for (body, maximum) in ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
        .iter()
        .zip(&maxima[2..])
    {
        assert!(*maximum <= 60.0, "{body}: {maximum} arcsec");
    }
    assert!(
        semidiameter_max <= 1.0,
        "Moon semidiameter: {semidiameter_max} arcsec"
    );

    let mut phase_max = 0.0_f64;
    let mut geocentric_semidiameter_max = 0.0_f64;
    for line in include_str!("data/horizons_vectors.dat")
        .lines()
        .filter(|line| line.starts_with("@moon_phase "))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let phase = moon_phase(parse_utc(fields[1])).unwrap();
        phase_max = phase_max
            .max((phase.illuminated_fraction - fields[2].parse::<f64>().unwrap() / 100.0).abs());
        geocentric_semidiameter_max = geocentric_semidiameter_max.max(
            (phase.apparent_semidiameter_arcsec - fields[3].parse::<f64>().unwrap() * 0.5).abs(),
        );
    }
    assert!(phase_max <= 0.005, "Moon phase fraction: {phase_max}");
    assert!(
        geocentric_semidiameter_max <= 1.0,
        "Moon geocentric semidiameter: {geocentric_semidiameter_max} arcsec"
    );

    let mut lunar_window_max = 0.0_f64;
    let mut lunar_window_rows = 0usize;
    let mut regression_present = false;
    for line in include_str!("data/horizons_vectors.dat")
        .lines()
        .filter(|line| line.starts_with("@moon_window "))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        lunar_window_rows += 1;
        regression_present |= fields[1] == "tromso" && fields[2] == "2008-11-14T00:00:00Z";
        let utc = parse_utc(fields[2]);
        let observer = Observer::new(
            Angle::<Degree>::new(fields[3].parse().unwrap()),
            Angle::new(fields[4].parse().unwrap()),
            fields[5].parse().unwrap(),
        )
        .unwrap();
        let actual = body_position(Body::Moon, utc, observer, false).unwrap();
        lunar_window_max = lunar_window_max.max(separation_arcsec(
            actual.azimuth.value(),
            actual.altitude.value(),
            fields[6].parse().unwrap(),
            fields[7].parse().unwrap(),
        ));
    }
    assert_eq!(lunar_window_rows, 621);
    assert!(regression_present);
    assert!(
        lunar_window_max <= 14.10,
        "Moon window: {lunar_window_max} arcsec"
    );
    println!("Moon 30-day window sweep max: {lunar_window_max} arcsec");
}

#[test]
fn delta_t_meets_declared_midmonth_residual() {
    let mut maximum = 0.0_f64;
    let mut rows = 0usize;
    for line in include_str!("data/horizons_vectors.dat")
        .lines()
        .filter(|line| line.starts_with("@delta_t_midmonth "))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        rows += 1;
        maximum = maximum.max(
            (delta_t_seconds(parse_utc(fields[1])).unwrap() - fields[2].parse::<f64>().unwrap())
                .abs(),
        );
    }
    assert_eq!(rows, 612);
    assert!(
        maximum <= 0.0066,
        "delta-T midpoint residual {maximum} seconds"
    );
}

#[test]
fn precession_and_lunar_parallax_are_load_bearing() {
    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap();
    let jd_tt = julian_day_tt(utc).unwrap();
    let observer = Observer::new(Angle::<Degree>::new(52.3676), Angle::new(4.9041), 0.0).unwrap();

    // Measured on the rendered catalog through the real reduction chain, as
    // DoD 5 states ("2026-epoch star positions"). A synthetic axis vector
    // would pass this gate even if `star_instances` skipped precession.
    let (min, median, max) =
        forge3d::astro::catalog::precession_ablation_arcminutes(utc, observer).unwrap();
    assert!(max > 20.0, "max star displacement {max} arcminutes");
    assert!(
        median > 15.0,
        "median star displacement {median} arcminutes"
    );
    // Precession rotates about the ecliptic pole, so some star must sit close
    // enough to that axis to be nearly unmoved. If the minimum were also large
    // the "ablation" would be measuring something other than precession.
    assert!(min < 5.0, "min star displacement {min} arcminutes");
    assert!(min <= median && median <= max);

    let sidereal = frames::gast(julian_day_ut1(utc).unwrap(), jd_tt);
    let lunar = moon::geocentric_ecliptic(jd_tt).unwrap();
    let (dpsi, deps, obliquity) = frames::nutation(jd_tt);
    let geocentric = DMat3::from_rotation_x(obliquity + deps)
        * DMat3::from_rotation_z(dpsi)
        * lunar.ecliptic_of_date_au;
    let topocentric = frames::geocentric_to_topocentric(geocentric, observer, sidereal);
    let geocentric_horizontal = frames::equatorial_to_horizontal(geocentric, observer, sidereal);
    let topocentric_horizontal = frames::equatorial_to_horizontal(topocentric, observer, sidereal);
    let parallax_arcminutes = separation_arcsec(
        geocentric_horizontal.0.value(),
        geocentric_horizontal.1.value(),
        topocentric_horizontal.0.value(),
        topocentric_horizontal.1.value(),
    ) / 60.0;
    assert!(
        parallax_arcminutes > 30.0,
        "{parallax_arcminutes} arcminutes"
    );
    println!(
        "ablation displacements arcminutes: precession over {} catalog stars \
         min={min:.3} median={median:.3} max={max:.3}, lunar_parallax={parallax_arcminutes:.3}",
        forge3d::astro::catalog::bright_star_catalog()
            .unwrap()
            .len()
    );
}

fn parse_utc(value: &str) -> UtcDateTime {
    UtcDateTime::new(
        value[0..4].parse().unwrap(),
        value[5..7].parse().unwrap(),
        value[8..10].parse().unwrap(),
        value[11..13].parse().unwrap(),
        value[14..16].parse().unwrap(),
        value[17..19].parse().unwrap(),
    )
    .unwrap()
}

fn body_index(body: Body) -> usize {
    match body {
        Body::Sun => 0,
        Body::Moon => 1,
        Body::Mercury => 2,
        Body::Venus => 3,
        Body::Mars => 4,
        Body::Jupiter => 5,
        Body::Saturn => 6,
    }
}

fn separation_arcsec(azimuth_a: f64, altitude_a: f64, azimuth_b: f64, altitude_b: f64) -> f64 {
    let azimuth_delta = (azimuth_a - azimuth_b).to_radians();
    let altitude_a = altitude_a.to_radians();
    let altitude_b = altitude_b.to_radians();
    let cosine = altitude_a.sin() * altitude_b.sin()
        + altitude_a.cos() * altitude_b.cos() * azimuth_delta.cos();
    cosine.clamp(-1.0, 1.0).acos().to_degrees() * 3_600.0
}
