//! Fixed spectral basis and CIE 1931 conversion for atmosphere integration.

use std::f32::consts::PI;

/// Eleven 40 nm bands spanning the visible range, inclusive.
pub const NUM_WAVELENGTHS: usize = 11;
pub const WAVELENGTHS_NM: [f32; NUM_WAVELENGTHS] = [
    380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0, 700.0, 740.0, 780.0,
];

/// CIE 1931 2-degree standard-observer colour matching functions sampled at
/// [`WAVELENGTHS_NM`]. Rows are x-bar, y-bar, z-bar.
pub const CIE_1931_XYZ: [[f32; 3]; NUM_WAVELENGTHS] = [
    [0.001_368, 0.000_039, 0.006_450],
    [0.134_380, 0.004_000, 0.645_600],
    [0.290_800, 0.060_000, 1.669_200],
    [0.004_900, 0.323_000, 0.272_000],
    [0.290_400, 0.954_000, 0.020_300],
    [0.916_300, 0.870_000, 0.001_650],
    [0.854_450, 0.381_000, 0.000_190],
    [0.164_900, 0.061_000, 0.000_000],
    [0.011_359, 0.004_102, 0.000_000],
    [0.000_690, 0.000_249, 0.000_000],
    [0.000_042, 0.000_015, 0.000_000],
];

pub const SEA_LEVEL_NUMBER_DENSITY_M3: f32 = 2.546_899e25;
const RAYLEIGH_CROSS_SECTION_550_M2: f32 = 5.10e-31;

const XYZ_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266, 1.876_010_8, 0.041_556],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MieParameters {
    pub extinction_550_m_inv: f32,
    pub single_scattering_albedo: f32,
    pub g: f32,
    pub angstrom_alpha: f32,
}

impl Default for MieParameters {
    fn default() -> Self {
        Self {
            extinction_550_m_inv: 2.0e-5,
            single_scattering_albedo: 0.9,
            g: 0.8,
            angstrom_alpha: 1.0,
        }
    }
}

impl MieParameters {
    pub fn validate(self) -> bool {
        self.extinction_550_m_inv.is_finite()
            && self.extinction_550_m_inv >= 0.0
            && self.single_scattering_albedo.is_finite()
            && (0.0..=1.0).contains(&self.single_scattering_albedo)
            && self.g.is_finite()
            && (-0.999..=0.999).contains(&self.g)
            && self.angstrom_alpha.is_finite()
            && (0.0..=4.0).contains(&self.angstrom_alpha)
    }

    pub fn extinction(self, wavelength_nm: f32) -> f32 {
        self.extinction_550_m_inv * (550.0 / wavelength_nm).powf(self.angstrom_alpha)
    }

    pub fn scattering(self, wavelength_nm: f32) -> f32 {
        self.extinction(wavelength_nm) * self.single_scattering_albedo
    }
}

pub fn rayleigh_scattering_cross_section(wavelength_nm: f32) -> f32 {
    assert!(wavelength_nm.is_finite() && wavelength_nm > 0.0);
    RAYLEIGH_CROSS_SECTION_550_M2 * (550.0 / wavelength_nm).powi(4)
}

pub fn rayleigh_scattering_coefficient(wavelength_nm: f32) -> f32 {
    rayleigh_scattering_cross_section(wavelength_nm) * SEA_LEVEL_NUMBER_DENSITY_M3
}

pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    3.0 * (1.0 + cos_theta.clamp(-1.0, 1.0).powi(2)) / (16.0 * PI)
}

pub fn mie_phase_cornette_shanks(cos_theta: f32, g: f32) -> f32 {
    let c = cos_theta.clamp(-1.0, 1.0);
    let g = g.clamp(-0.999, 0.999);
    let denominator = (1.0 + g * g - 2.0 * g * c).max(1.0e-6).powf(1.5);
    3.0 * (1.0 - g * g) * (1.0 + c * c) / (8.0 * PI * (2.0 + g * g) * denominator)
}

fn integrate_xyz(samples: &[f32]) -> [f32; 3] {
    let mut xyz = [0.0; 3];
    for (i, sample) in samples.iter().copied().enumerate() {
        let weight = if i == 0 || i + 1 == NUM_WAVELENGTHS {
            0.5
        } else {
            1.0
        };
        for (component, value) in xyz.iter_mut().enumerate() {
            *value += sample * CIE_1931_XYZ[i][component] * weight;
        }
    }
    xyz
}

fn xyz_to_rgb(xyz: [f32; 3]) -> [f32; 3] {
    let mut rgb = [0.0; 3];
    for row in 0..3 {
        rgb[row] = XYZ_TO_LINEAR_SRGB[row][0] * xyz[0]
            + XYZ_TO_LINEAR_SRGB[row][1] * xyz[1]
            + XYZ_TO_LINEAR_SRGB[row][2] * xyz[2];
    }
    rgb
}

pub fn spectral_to_linear_rgb(samples: &[f32]) -> [f32; 3] {
    assert_eq!(samples.len(), NUM_WAVELENGTHS);
    let raw = xyz_to_rgb(integrate_xyz(samples));
    let white = xyz_to_rgb(integrate_xyz(&[1.0; NUM_WAVELENGTHS]));
    [raw[0] / white[0], raw[1] / white[1], raw[2] / white[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_spectrum_is_neutral() {
        let rgb = spectral_to_linear_rgb(&[1.0; NUM_WAVELENGTHS]);
        assert!(rgb.iter().all(|v| v.is_finite()));
        assert!((rgb[0] - 1.0).abs() < 1.0e-5);
        assert!((rgb[0] - rgb[1]).abs() < 1.0e-5);
        assert!((rgb[1] - rgb[2]).abs() < 1.0e-5);
    }

    #[test]
    fn rayleigh_retains_lambda_to_minus_four() {
        let short = rayleigh_scattering_cross_section(400.0);
        let long = rayleigh_scattering_cross_section(800.0);
        assert!((short / long - 16.0).abs() < 2.0e-5);
    }

    #[test]
    fn phase_functions_are_finite_and_nonnegative() {
        for i in 0..=100 {
            let cosine = -1.0 + i as f32 * 0.02;
            assert!(rayleigh_phase(cosine).is_finite());
            assert!(mie_phase_cornette_shanks(cosine, 0.8) >= 0.0);
        }
    }
}
