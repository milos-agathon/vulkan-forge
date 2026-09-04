//! GPU instances and directional moonlight for SIDERA's night renderer.
//!
//! Lunar attenuation follows Krisciunas & Schaefer (1991), PASP 103, 1033:
//! the phase polynomial from their equation 9 and a standard 0.172 mag/airmass
//! extinction. The renderer divides by the modeled post-extinction zenith full
//! Moon so forge3d's relative directional-light intensity reaches one there.

use super::{catalog, observation::ObservationSnapshot, Body};
use glam::DVec3;

pub const MAX_NIGHT_INSTANCES: usize = 9_096 + 6;
const FULL_MOON_LUX: f64 = 0.25;
/// Pixel-integrated tent support for a sub-pixel point source. The shader uses
/// this radius only to rasterize the four pixels that can receive energy; its
/// separable tent weights sum to one for every sub-pixel star position, so
/// changing output resolution cannot change integrated display energy.
const STAR_SPLAT_RADIUS_PX: f32 = 1.0;
const STAR_EXPOSURE: f32 = 0.20;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NightInstance {
    pub direction_radius: [f32; 4],
    pub right_kind: [f32; 4],
    pub up_phase: [f32; 4],
    pub color_intensity: [f32; 4],
}

pub fn instances(observation: &ObservationSnapshot) -> Vec<NightInstance> {
    let mut result = Vec::with_capacity(MAX_NIGHT_INSTANCES);
    let sun = observation.bodies[0].1;
    let night_visibility = catalog::twilight_visibility(sun.altitude) as f32;
    for star in &observation.stars {
        let direction = horizontal_direction(star.azimuth.value(), star.altitude.value());
        let (right, up) = tangent_basis(direction, None);
        // Irradiance relative to a magnitude-0 star. Taken from the catalog's
        // physical value rather than recomputed, so the cited photometry has
        // exactly one definition in the tree.
        let relative_irradiance =
            (star.irradiance_w_m2 / catalog::magnitude_to_irradiance(0.0)) as f32;
        result.push(instance(
            direction,
            right,
            up,
            STAR_SPLAT_RADIUS_PX,
            0.0,
            night_visibility,
            star.linear_rgb,
            star_display_energy(relative_irradiance),
        ));
    }

    // Deliberately artistic display constants: SIDERA makes no physical
    // planetary-photometry claim. This assumption is declared in the signed
    // render-certificate model ledger by the night rendering path.
    const PLANETS: [(Body, [f32; 3], f32); 5] = [
        (Body::Mercury, [0.75, 0.72, 0.68], 0.55),
        (Body::Venus, [1.0, 0.92, 0.72], 1.0),
        (Body::Mars, [1.0, 0.32, 0.16], 0.65),
        (Body::Jupiter, [0.95, 0.82, 0.62], 0.85),
        (Body::Saturn, [0.95, 0.82, 0.52], 0.7),
    ];
    for (body, color, intensity) in PLANETS {
        let position = observation
            .bodies
            .iter()
            .find(|(candidate, _)| *candidate == body)
            .expect("observation contains every rendered planet")
            .1;
        let direction = horizontal_direction(position.azimuth.value(), position.altitude.value());
        let (right, up) = tangent_basis(direction, None);
        result.push(instance(
            direction,
            right,
            up,
            0.06_f32.to_radians(),
            1.0,
            night_visibility,
            color,
            intensity,
        ));
    }

    let moon = observation.bodies[1].1;
    let sun_direction = horizontal_direction(sun.azimuth.value(), sun.altitude.value());
    let moon_direction = horizontal_direction(moon.azimuth.value(), moon.altitude.value());
    let bright_limb =
        (sun_direction - moon_direction * sun_direction.dot(moon_direction)).try_normalize();
    let (right, up) = tangent_basis(moon_direction, bright_limb);
    result.push(instance(
        moon_direction,
        right,
        up,
        (observation.moon_phase.apparent_semidiameter_arcsec / 3_600.0).to_radians() as f32,
        2.0,
        observation.moon_phase.phase_angle.radians() as f32,
        [0.92, 0.92, 0.9],
        // The Moon is independent of the star-only twilight visibility ramp.
        // Its direction is still horizon-clipped in the vertex shader.
        1.0,
    ));
    result
}

fn star_display_energy(relative_irradiance: f32) -> f32 {
    relative_irradiance.max(0.0) * STAR_EXPOSURE
}

/// K&S phase attenuation plus standard-atmosphere extinction, in lux.
pub fn moon_illuminance_lux(phase_angle_deg: f64, altitude_deg: f64) -> f64 {
    if altitude_deg <= 0.0 {
        return 0.0;
    }
    let phase = phase_angle_deg.abs().clamp(0.0, 180.0);
    let phase_attenuation = 10.0_f64.powf(-0.4 * (0.026 * phase + 4.0e-9 * phase.powi(4)));
    let zenith = (90.0 - altitude_deg.clamp(0.0, 90.0)).to_radians();
    let airmass = (1.0 - 0.96 * zenith.sin().powi(2)).powf(-0.5);
    FULL_MOON_LUX * phase_attenuation * 10.0_f64.powf(-0.4 * 0.172 * airmass)
}

pub fn normalized_moonlight(phase_angle_deg: f64, altitude_deg: f64) -> f32 {
    let modeled_zenith_full_moon = moon_illuminance_lux(0.0, 90.0);
    (moon_illuminance_lux(phase_angle_deg, altitude_deg) / modeled_zenith_full_moon) as f32
}

pub fn horizontal_direction(azimuth_deg: f64, altitude_deg: f64) -> DVec3 {
    let azimuth = azimuth_deg.to_radians();
    let altitude = altitude_deg.to_radians();
    DVec3::new(
        altitude.cos() * azimuth.sin(),
        altitude.sin(),
        -altitude.cos() * azimuth.cos(),
    )
}

fn tangent_basis(direction: DVec3, preferred_right: Option<DVec3>) -> (DVec3, DVec3) {
    let right = preferred_right.unwrap_or_else(|| {
        DVec3::Y
            .cross(direction)
            .try_normalize()
            .unwrap_or(DVec3::X)
    });
    (right, direction.cross(right).normalize())
}

#[allow(clippy::too_many_arguments)]
fn instance(
    direction: DVec3,
    right: DVec3,
    up: DVec3,
    radius: f32,
    kind: f32,
    phase: f32,
    color: [f32; 3],
    intensity: f32,
) -> NightInstance {
    NightInstance {
        direction_radius: [
            direction.x as f32,
            direction.y as f32,
            direction.z as f32,
            radius,
        ],
        right_kind: [right.x as f32, right.y as f32, right.z as f32, kind],
        up_phase: [up.x as f32, up.y as f32, up.z as f32, phase],
        color_intensity: [color[0], color[1], color[2], intensity],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{astro::time::UtcDateTime, geo::units::Angle};

    #[test]
    fn krisciunas_schaefer_phase_and_horizon_are_load_bearing() {
        let full = moon_illuminance_lux(0.0, 90.0);
        let quarter = moon_illuminance_lux(90.0, 90.0);
        let crescent = moon_illuminance_lux(150.0, 90.0);
        assert!(full > 0.20 && full < 0.23);
        assert!(full > quarter * 8.0);
        assert!(quarter > crescent * 4.0);
        assert_eq!(moon_illuminance_lux(0.0, -0.1), 0.0);
        assert!((normalized_moonlight(0.0, 90.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn geographic_azimuth_uses_the_viewer_east_up_negative_north_basis() {
        let north = horizontal_direction(0.0, 0.0);
        let east = horizontal_direction(90.0, 0.0);
        let south = horizontal_direction(180.0, 0.0);
        assert!(north.abs_diff_eq(DVec3::NEG_Z, 1.0e-12));
        assert!(east.abs_diff_eq(DVec3::X, 1.0e-12));
        assert!(south.abs_diff_eq(DVec3::Z, 1.0e-12));
    }

    #[test]
    fn declared_twilight_ramp_hides_catalog_objects_by_day() {
        let utc = UtcDateTime::new(2026, 12, 27, 22, 0, 0.0).unwrap();
        let observer =
            super::super::Observer::new(Angle::new(19.8207), Angle::new(-155.4681), 4_205.0)
                .unwrap();
        let observation = super::super::observation::set_observation(utc, observer).unwrap();
        let upload = instances(&observation);
        assert!(observation.bodies[0].1.altitude.value() > -4.0);
        // Stars and planets carry the ramp in `up_phase[3]`.
        assert!(upload[..MAX_NIGHT_INSTANCES - 1]
            .iter()
            .all(|item| item.up_phase[3] == 0.0));
        // The Moon is explicitly not governed by the star-only twilight ramp.
        let moon = upload.last().unwrap();
        assert_eq!(moon.right_kind[3], 2.0);
        assert_eq!(moon.color_intensity[3], 1.0);
    }

    #[test]
    fn night_instance_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<NightInstance>(), 64);
        assert_eq!(MAX_NIGHT_INSTANCES, 9_102);
    }

    #[test]
    fn star_display_energy_preserves_the_magnitude_ratio() {
        let magnitude_zero = catalog::magnitude_to_irradiance(0.0);
        let magnitude_five = catalog::magnitude_to_irradiance(5.0);
        assert!((magnitude_zero / magnitude_five - 100.0).abs() < 1.0e-12);

        let zero_energy = star_display_energy(1.0);
        let five_energy = star_display_energy((magnitude_five / magnitude_zero) as f32);
        assert!((zero_energy / five_energy - 100.0).abs() < 1.0e-3);

        // V=-1.46 (Sirius) remains below saturation in the linear HDR target.
        let sirius_relative = 10.0_f32.powf(0.4 * 1.46);
        assert!(star_display_energy(sirius_relative) < 1.0);

        // A V=6.5 star survives the HDR blend and changes a byte after the one
        // authoritative Reinhard + sRGB display resolve; no visibility floor
        // or gamma-encoded alpha is needed.
        let limit_alpha = star_display_energy(10.0_f32.powf(-0.4 * 6.5));
        let background = [0.004, 0.006, 0.018, 1.0];
        let blended = [
            1.0 * limit_alpha + background[0] * (1.0 - limit_alpha),
            1.0 * limit_alpha + background[1] * (1.0 - limit_alpha),
            1.0 * limit_alpha + background[2] * (1.0 - limit_alpha),
            1.0,
        ];
        let before = crate::core::tonemap::resolve_reference_hdr_to_rgba8(&background, 1.0);
        let after = crate::core::tonemap::resolve_reference_hdr_to_rgba8(&blended, 1.0);
        assert_ne!(before[..3], after[..3]);
    }

    #[test]
    fn star_shader_uses_direct_energy_without_a_visibility_floor() {
        let shader = include_str!("../shaders/stars.wgsl");
        assert!(shader.contains("clamp(input.color_intensity.w, 0.0, 1.0)"));
        assert!(!shader.contains("0.18 + 0.65 * sqrt"));
    }

    #[test]
    fn fixed_golden_upload_contains_catalog_planets_and_oriented_moon() {
        let utc = UtcDateTime::new(2026, 12, 27, 14, 0, 0.0).unwrap();
        let observer =
            super::super::Observer::new(Angle::new(19.8207), Angle::new(-155.4681), 4_205.0)
                .unwrap();
        let observation = super::super::observation::set_observation(utc, observer).unwrap();
        let upload = instances(&observation);
        assert_eq!(upload.len(), MAX_NIGHT_INSTANCES);
        assert_eq!(
            upload
                .iter()
                .filter(|item| item.right_kind[3] == 1.0)
                .count(),
            5
        );
        let moon = upload.last().unwrap();
        assert_eq!(moon.right_kind[3], 2.0);
        let geometric_fraction = (1.0 + f64::from(moon.up_phase[3]).cos()) * 0.5;
        assert!((geometric_fraction - observation.moon_phase.illuminated_fraction).abs() < 1e-6);
        let moon_direction = DVec3::new(
            moon.direction_radius[0] as f64,
            moon.direction_radius[1] as f64,
            moon.direction_radius[2] as f64,
        );
        let right = DVec3::new(
            moon.right_kind[0] as f64,
            moon.right_kind[1] as f64,
            moon.right_kind[2] as f64,
        );
        let sun = observation.bodies[0].1;
        let sun_direction = horizontal_direction(sun.azimuth.value(), sun.altitude.value());
        let projected_sun =
            (sun_direction - moon_direction * sun_direction.dot(moon_direction)).normalize();
        assert!(right.dot(projected_sun) > 0.999_999);
    }
}
