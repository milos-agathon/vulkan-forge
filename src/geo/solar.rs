//! NREL Solar Position Algorithm (SPA), Reda & Andreas (2003).

use super::solar_coefficients::{
    NUTATION_COEFFS, OBLIQUITY_COEFFS, TERMS_B, TERMS_L, TERMS_PE, TERMS_R, TERMS_Y,
};

const ABERRATION_ARCSEC: f64 = -20.4898;
const EARTH_FLATTENING_FACTOR: f64 = 0.99664719;
const EARTH_RADIUS_M: f64 = 6_378_140.0;

#[derive(Debug, Clone, Copy)]
pub struct SolarTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: f64,
    pub tz_offset_hours: f64,
    pub delta_t_seconds: f64,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub elevation_m: f64,
    pub pressure_mbar: f64,
    pub temperature_c: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SolarVector {
    pub zenith_deg: f64,
    pub azimuth_deg: f64,
    pub apparent_elevation_deg: f64,
    pub true_elevation_deg: f64,
    pub distance_au: f64,
    pub equation_of_time_min: f64,
}

#[derive(Debug, Clone, Copy)]
struct JulianDate {
    jd: f64,
    jc: f64,
    jce: f64,
    jme: f64,
}

impl SolarTime {
    pub fn validate(&self) -> Result<(), String> {
        if !(-2000..=6000).contains(&self.year) {
            return Err("year must be in [-2000, 6000]".into());
        }
        if !(1..=12).contains(&self.month)
            || self.day == 0
            || self.day > days_in_month(self.year, self.month)
            || self.hour > 23
            || self.minute > 59
            || !self.second.is_finite()
            || !(0.0..60.0).contains(&self.second)
        {
            return Err("invalid civil date/time".into());
        }
        finite_range("latitude", self.latitude_deg, -90.0, 90.0)?;
        finite_range("longitude", self.longitude_deg, -180.0, 180.0)?;
        finite_range("tz_offset_hours", self.tz_offset_hours, -18.0, 18.0)?;
        if !self.delta_t_seconds.is_finite() || !self.elevation_m.is_finite() {
            return Err("delta_t_seconds and elevation_m must be finite".into());
        }
        if !self.pressure_mbar.is_finite() || self.pressure_mbar <= 0.0 {
            return Err("pressure_mbar must be finite and positive".into());
        }
        if !self.temperature_c.is_finite() || self.temperature_c <= -273.15 {
            return Err("temperature_c must be above absolute zero".into());
        }
        Ok(())
    }
}

pub fn solar_position(time: &SolarTime) -> Result<SolarVector, String> {
    time.validate()?;
    let jd = julian_date(time);
    let l = lbr_degrees(jd.jme, TERMS_L);
    let b = lbr_degrees(jd.jme, TERMS_B);
    let r = lbr_polynomial(jd.jme, TERMS_R);
    let theta = normalize_degrees(l + 180.0);
    let beta = -b;

    let x = nutation_terms(jd.jce);
    let (delta_psi, delta_epsilon) = nutation(jd.jce, &x);
    let epsilon0 = polynomial(OBLIQUITY_COEFFS, jd.jme / 10.0);
    let epsilon = epsilon0 / 3600.0 + delta_epsilon;
    let lambda = theta + delta_psi + ABERRATION_ARCSEC / (3600.0 * r);
    let nu0 = normalize_degrees(
        280.46061837
            + 360.98564736629 * (jd.jd - 2451545.0)
            + jd.jc * jd.jc * (0.000387933 - jd.jc / 38710000.0),
    );
    let nu = nu0 + delta_psi * epsilon.to_radians().cos();

    let beta_rad = beta.to_radians();
    let epsilon_rad = epsilon.to_radians();
    let lambda_rad = lambda.to_radians();
    let alpha = normalize_degrees(
        (lambda_rad.sin() * epsilon_rad.cos() - beta_rad.tan() * epsilon_rad.sin())
            .atan2(lambda_rad.cos())
            .to_degrees(),
    );
    let delta = (beta_rad.sin() * epsilon_rad.cos()
        + beta_rad.cos() * epsilon_rad.sin() * lambda_rad.sin())
    .asin()
    .to_degrees();

    let h = normalize_degrees(nu + time.longitude_deg - alpha);
    let xi = (8.794 / (3600.0 * r)).to_radians();
    let phi = time.latitude_deg.to_radians();
    let delta_rad = delta.to_radians();
    let h_rad = h.to_radians();
    let u = (EARTH_FLATTENING_FACTOR * phi.tan()).atan();
    let observer_y =
        EARTH_FLATTENING_FACTOR * u.sin() + (time.elevation_m / EARTH_RADIUS_M) * phi.sin();
    let observer_x = u.cos() + (time.elevation_m / EARTH_RADIUS_M) * phi.cos();
    let delta_alpha = (-observer_x * xi.sin() * h_rad.sin())
        .atan2(delta_rad.cos() - observer_x * xi.sin() * h_rad.cos())
        .to_degrees();
    let delta_prime = ((delta_rad.sin() - observer_y * xi.sin()) * delta_alpha.to_radians().cos())
        .atan2(delta_rad.cos() - observer_x * xi.sin() * h_rad.cos());
    let h_prime = (h - delta_alpha).to_radians();
    let true_zenith = (phi.sin() * delta_prime.sin()
        + phi.cos() * delta_prime.cos() * h_prime.cos())
    .clamp(-1.0, 1.0)
    .acos()
    .to_degrees();
    let true_elevation = 90.0 - true_zenith;
    let refraction = if true_elevation >= -0.83337 {
        (time.pressure_mbar / 1010.0) * (283.0 / (273.0 + time.temperature_c)) * 1.02
            / (60.0
                * (true_elevation + 10.3 / (true_elevation + 5.11))
                    .to_radians()
                    .tan())
    } else {
        0.0
    };
    let apparent_elevation = true_elevation + refraction;
    let azimuth = normalize_degrees(
        180.0
            + h_prime
                .sin()
                .atan2(h_prime.cos() * phi.sin() - delta_prime.tan() * phi.cos())
                .to_degrees(),
    );
    let mean_longitude = normalize_degrees(
        280.4664567
            + jd.jme
                * (360007.6982779
                    + jd.jme
                        * (0.03032028
                            + jd.jme
                                * (1.0 / 49931.0
                                    + jd.jme * (-1.0 / 15300.0 - jd.jme / 2000000.0)))),
    );
    let mut equation_of_time =
        4.0 * (mean_longitude - 0.0057183 - alpha + delta_psi * epsilon_rad.cos());
    equation_of_time = equation_of_time.rem_euclid(1440.0);
    if equation_of_time > 20.0 {
        equation_of_time -= 1440.0;
    }

    Ok(SolarVector {
        zenith_deg: 90.0 - apparent_elevation,
        azimuth_deg: azimuth,
        apparent_elevation_deg: apparent_elevation,
        true_elevation_deg: true_elevation,
        distance_au: r,
        equation_of_time_min: equation_of_time,
    })
}

fn julian_date(time: &SolarTime) -> JulianDate {
    let mut year = time.year;
    let mut month = time.month as i32;
    if month <= 2 {
        year -= 1;
        month += 12;
    }
    let day = time.day as f64
        + (time.hour as f64 + time.minute as f64 / 60.0 + time.second / 3600.0
            - time.tz_offset_hours)
            / 24.0;
    let a = (year as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd = (365.25 * (year as f64 + 4716.0)).floor()
        + (30.6001 * (month as f64 + 1.0)).floor()
        + day
        + b
        - 1524.5;
    let jde = jd + time.delta_t_seconds / 86400.0;
    let jc = (jd - 2451545.0) / 36525.0;
    let jce = (jde - 2451545.0) / 36525.0;
    JulianDate {
        jd,
        jc,
        jce,
        jme: jce / 10.0,
    }
}

fn lbr_polynomial(jme: f64, term_sets: &[&[&[f64; 3]]]) -> f64 {
    let mut sums = [0.0; 6];
    for (i, terms) in term_sets.iter().enumerate() {
        sums[i] = terms
            .iter()
            .map(|term| term[0] * (term[1] + term[2] * jme).cos())
            .sum();
    }
    polynomial(&sums[..term_sets.len()], jme) / 1e8
}

fn lbr_degrees(jme: f64, terms: &[&[&[f64; 3]]]) -> f64 {
    normalize_degrees(lbr_polynomial(jme, terms).to_degrees())
}

fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value.mul_add(x, *coefficient))
}

fn nutation_terms(jce: f64) -> [f64; 5] {
    [
        polynomial(NUTATION_COEFFS[0], jce),
        polynomial(NUTATION_COEFFS[1], jce),
        polynomial(NUTATION_COEFFS[2], jce),
        polynomial(NUTATION_COEFFS[3], jce),
        polynomial(NUTATION_COEFFS[4], jce),
    ]
}

fn nutation(jce: f64, x: &[f64; 5]) -> (f64, f64) {
    let mut psi = 0.0;
    let mut epsilon = 0.0;
    for (i, pe) in TERMS_PE.iter().enumerate() {
        let argument = x
            .iter()
            .enumerate()
            .map(|(j, value)| value * f64::from(TERMS_Y[i][j]))
            .sum::<f64>()
            .to_radians();
        psi += (pe[0] + pe[1] * jce) * argument.sin();
        epsilon += (pe[2] + pe[3] * jce) * argument.cos();
    }
    (psi / 36_000_000.0, epsilon / 36_000_000.0)
}

fn normalize_degrees(value: f64) -> f64 {
    value.rem_euclid(360.0)
}

fn finite_range(name: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(format!("{name} must be finite and in [{min}, {max}]"))
    }
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nrel_worked_example() {
        let result = solar_position(&SolarTime {
            year: 2003,
            month: 10,
            day: 17,
            hour: 12,
            minute: 30,
            second: 30.0,
            tz_offset_hours: -7.0,
            delta_t_seconds: 67.0,
            latitude_deg: 39.742476,
            longitude_deg: -105.1786,
            elevation_m: 1830.14,
            pressure_mbar: 820.0,
            temperature_c: 11.0,
        })
        .unwrap();
        assert!((result.zenith_deg - 50.11162).abs() < 0.0003);
        assert!((result.azimuth_deg - 194.34024).abs() < 0.0003);
    }
}
