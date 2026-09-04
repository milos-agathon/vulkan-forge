//! Deterministic CPU atmosphere LUT generation and independent reference integration.

use super::precomputed::{self, TURBIDITY_BANK};
use super::spectral::{
    mie_phase_cornette_shanks, rayleigh_phase, rayleigh_scattering_coefficient,
    spectral_to_linear_rgb, MieParameters, NUM_WAVELENGTHS, WAVELENGTHS_NM,
};
use crate::core::resource_tracker::{tracked_host_allocation, ResourceHandle};
use half::f16;
use sha2::{Digest, Sha256};

const RGBA_CHANNELS: usize = 4;
const HOST_VISIBLE_LIMIT_BYTES: u64 = 512 * 1024 * 1024;
const PRECOMPUTED_OZONE_DU: f32 = 300.0;
const PRECOMPUTED_MIE_G: f32 = 0.8;

#[derive(Debug, thiserror::Error)]
pub enum AtmosphereError {
    #[error("invalid atmosphere configuration: {0}")]
    InvalidConfig(String),
    #[error("atmosphere LUT dimensions overflow")]
    DimensionOverflow,
    #[error("atmosphere LUT payload requires {requested} bytes, over the {limit} byte limit")]
    BudgetExceeded { requested: u64, limit: u64 },
    #[error("host-visible atmosphere LUT budget rejected: {0}")]
    TrackerBudget(String),
    #[error("precomputed atmosphere bank does not support {0}")]
    UnsupportedPrecomputedConfig(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LutDimensions {
    pub transmittance_mu: u32,
    pub transmittance_height: u32,
    pub scattering_mu_view: u32,
    pub scattering_mu_sun: u32,
    pub scattering_height: u32,
    pub scattering_nu: u32,
    pub aerial_distance: u32,
    pub aerial_mu_view: u32,
    pub aerial_height: u32,
}

impl Default for LutDimensions {
    fn default() -> Self {
        Self {
            transmittance_mu: 32,
            transmittance_height: 8,
            scattering_mu_view: 17,
            scattering_mu_sun: 17,
            scattering_height: 8,
            scattering_nu: 16,
            aerial_distance: 8,
            aerial_mu_view: 8,
            aerial_height: 8,
        }
    }
}

impl LutDimensions {
    const fn precomputed() -> Self {
        Self {
            transmittance_mu: precomputed::TRANSMITTANCE_DIMENSIONS[0],
            transmittance_height: precomputed::TRANSMITTANCE_DIMENSIONS[1],
            scattering_mu_view: precomputed::SCATTERING_DIMENSIONS[0],
            scattering_mu_sun: precomputed::SCATTERING_DIMENSIONS[1],
            scattering_height: precomputed::SCATTERING_HEIGHT,
            scattering_nu: precomputed::SCATTERING_NU,
            aerial_distance: precomputed::AERIAL_DIMENSIONS[0],
            aerial_mu_view: precomputed::AERIAL_DIMENSIONS[1],
            aerial_height: precomputed::AERIAL_DIMENSIONS[2],
        }
    }

    fn validate(self) -> Result<(), AtmosphereError> {
        let axes = [
            self.transmittance_mu,
            self.transmittance_height,
            self.scattering_mu_view,
            self.scattering_mu_sun,
            self.scattering_height,
            self.scattering_nu,
            self.aerial_distance,
            self.aerial_mu_view,
            self.aerial_height,
        ];
        if axes.iter().any(|&axis| axis < 2) {
            return Err(AtmosphereError::InvalidConfig(
                "every atmosphere LUT axis must contain at least two samples".into(),
            ));
        }
        if axes.iter().any(|&axis| axis > 256) {
            return Err(AtmosphereError::InvalidConfig(
                "atmosphere LUT axes are capped at 256 samples".into(),
            ));
        }
        Ok(())
    }

    pub fn texel_count(self) -> Result<u64, AtmosphereError> {
        self.validate()?;
        let transmittance = u64::from(self.transmittance_mu)
            .checked_mul(u64::from(self.transmittance_height))
            .ok_or(AtmosphereError::DimensionOverflow)?;
        let scattering = u64::from(self.scattering_mu_view)
            .checked_mul(u64::from(self.scattering_mu_sun))
            .and_then(|n| n.checked_mul(u64::from(self.scattering_height)))
            .and_then(|n| n.checked_mul(u64::from(self.scattering_nu)))
            .ok_or(AtmosphereError::DimensionOverflow)?;
        let aerial = u64::from(self.aerial_distance)
            .checked_mul(u64::from(self.aerial_mu_view))
            .and_then(|n| n.checked_mul(u64::from(self.aerial_height)))
            .ok_or(AtmosphereError::DimensionOverflow)?;
        transmittance
            .checked_add(
                scattering
                    .checked_mul(2)
                    .ok_or(AtmosphereError::DimensionOverflow)?,
            )
            .and_then(|n| n.checked_add(aerial))
            .ok_or(AtmosphereError::DimensionOverflow)
    }

    pub fn payload_bytes(self) -> Result<u64, AtmosphereError> {
        self.texel_count()?
            .checked_mul((RGBA_CHANNELS * std::mem::size_of::<f16>()) as u64)
            .ok_or(AtmosphereError::DimensionOverflow)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtmosphereConfig {
    pub turbidity: f32,
    pub ozone_du: f32,
    pub mie_g: f32,
    pub bottom_radius_m: f32,
    pub top_radius_m: f32,
    pub rayleigh_scale_height_m: f32,
    pub mie_scale_height_m: f32,
    pub max_aerial_distance_m: f32,
    pub ground_albedo: f32,
    pub scattering_orders: u32,
    pub dimensions: LutDimensions,
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            turbidity: 2.0,
            ozone_du: PRECOMPUTED_OZONE_DU,
            mie_g: PRECOMPUTED_MIE_G,
            bottom_radius_m: 6_360_000.0,
            top_radius_m: 6_460_000.0,
            rayleigh_scale_height_m: 8_000.0,
            mie_scale_height_m: 1_200.0,
            max_aerial_distance_m: 160_000.0,
            ground_albedo: 0.3,
            scattering_orders: 4,
            dimensions: LutDimensions::default(),
        }
    }
}

impl AtmosphereConfig {
    pub fn validate(&self) -> Result<(), AtmosphereError> {
        let finite = [
            self.turbidity,
            self.ozone_du,
            self.mie_g,
            self.bottom_radius_m,
            self.top_radius_m,
            self.rayleigh_scale_height_m,
            self.mie_scale_height_m,
            self.max_aerial_distance_m,
            self.ground_albedo,
        ];
        if finite.iter().any(|v| !v.is_finite()) {
            return Err(AtmosphereError::InvalidConfig(
                "all scalar parameters must be finite".into(),
            ));
        }
        if !(1.0..=10.0).contains(&self.turbidity) {
            return Err(AtmosphereError::InvalidConfig(
                "turbidity must be in [1, 10]".into(),
            ));
        }
        if !(0.0..=600.0).contains(&self.ozone_du) {
            return Err(AtmosphereError::InvalidConfig(
                "ozone must be in [0, 600] DU".into(),
            ));
        }
        if !(0.0..=0.99).contains(&self.mie_g) {
            return Err(AtmosphereError::InvalidConfig(
                "mie_g must be in [0, 0.99]".into(),
            ));
        }
        if self.bottom_radius_m <= 0.0 || self.top_radius_m <= self.bottom_radius_m {
            return Err(AtmosphereError::InvalidConfig(
                "top radius must exceed a positive bottom radius".into(),
            ));
        }
        if self.rayleigh_scale_height_m <= 0.0
            || self.mie_scale_height_m <= 0.0
            || self.max_aerial_distance_m <= 0.0
        {
            return Err(AtmosphereError::InvalidConfig(
                "scale heights and aerial distance must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.ground_albedo) {
            return Err(AtmosphereError::InvalidConfig(
                "ground albedo must be in [0, 1]".into(),
            ));
        }
        if !(2..=8).contains(&self.scattering_orders) {
            return Err(AtmosphereError::InvalidConfig(
                "scattering_orders must be in [2, 8]".into(),
            ));
        }
        self.dimensions.validate()?;
        let bytes = self.dimensions.payload_bytes()?;
        if bytes > HOST_VISIBLE_LIMIT_BYTES {
            return Err(AtmosphereError::BudgetExceeded {
                requested: bytes,
                limit: HOST_VISIBLE_LIMIT_BYTES,
            });
        }
        Ok(())
    }

    fn atmosphere_height_m(&self) -> f32 {
        self.top_radius_m - self.bottom_radius_m
    }
    fn mie(&self) -> MieParameters {
        MieParameters {
            extinction_550_m_inv: 1.0e-5 * self.turbidity,
            single_scattering_albedo: 0.9,
            g: self.mie_g,
            angstrom_alpha: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtmosphereLutMetadata {
    pub(crate) config: AtmosphereConfig,
    pub(crate) dimensions: LutDimensions,
    pub(crate) wavelengths_nm: [f32; NUM_WAVELENGTHS],
    pub(crate) scattering_orders: u32,
    pub(crate) scattering_lut_semantics: &'static str,
    pub(crate) storage_format: &'static str,
    pub(crate) precomputed: bool,
    pub(crate) precomputed_turbidity_bracket: Option<[f32; 2]>,
}

impl AtmosphereLutMetadata {
    pub fn config(&self) -> &AtmosphereConfig {
        &self.config
    }
    pub fn dimensions(&self) -> LutDimensions {
        self.dimensions
    }
    pub fn wavelengths_nm(&self) -> &[f32; NUM_WAVELENGTHS] {
        &self.wavelengths_nm
    }
    pub fn scattering_orders(&self) -> u32 {
        self.scattering_orders
    }
    pub fn scattering_lut_semantics(&self) -> &'static str {
        self.scattering_lut_semantics
    }
    pub fn storage_format(&self) -> &'static str {
        self.storage_format
    }
    pub fn is_precomputed(&self) -> bool {
        self.precomputed
    }
    pub fn precomputed_turbidity_bracket(&self) -> Option<[f32; 2]> {
        self.precomputed_turbidity_bracket
    }
}

pub const ACCUMULATED_SCATTERING_LUT_SEMANTICS: &str =
    "accumulated-single-plus-higher-orders-density-height-u-squared";
/// The optional aerial volume stores no radiance shortcut: RGB is exactly zero
/// and alpha is the mean finite-segment spectral transmittance. Active terrain
/// and PROMETHEUS paths derive in-scatter from the accumulated-scattering LUT.
pub const AERIAL_TRANSMITTANCE_LUT_SEMANTICS: &str = "rgb-zero-alpha-mean-segment-transmittance";

#[cfg(any(feature = "atmosphere-bake", test))]
fn scattering_mu_from_unit(unit: f32) -> f32 {
    let x = 2.0 * unit.clamp(0.0, 1.0) - 1.0;
    x.signum() * x.abs().powi(2)
}
fn scattering_mu_to_unit(mu: f32) -> f32 {
    let mu = mu.clamp(-1.0, 1.0);
    (mu.signum() * mu.abs().sqrt() + 1.0) * 0.5
}
#[cfg(any(feature = "atmosphere-bake", test))]
fn scattering_nu_from_unit(unit: f32) -> f32 {
    let d = 1.0 - unit.clamp(0.0, 1.0);
    1.0 - 2.0 * d * d
}
fn scattering_nu_to_unit(nu: f32) -> f32 {
    1.0 - ((1.0 - nu.clamp(-1.0, 1.0)) * 0.5).sqrt()
}

/// Density-aware altitude coordinate for the accumulated-scattering LUT.
/// Eight linear slices are 14.3 km apart and cannot resolve the 1.2 km Mie
/// layer. Storing h = H*u^2 keeps the existing dimensions and concentrates
/// slices where both atmospheric density profiles vary fastest.
fn scattering_height_to_unit(height_m: f32, atmosphere_height_m: f32) -> f32 {
    (height_m.clamp(0.0, atmosphere_height_m) / atmosphere_height_m).sqrt()
}

#[cfg(any(feature = "atmosphere-bake", test))]
fn scattering_height_from_unit(unit: f32, atmosphere_height_m: f32) -> f32 {
    atmosphere_height_m * unit.clamp(0.0, 1.0).powi(2)
}

#[derive(Debug)]
pub struct LutData {
    pub(crate) dimensions: [u32; 3],
    pub(crate) texels: Vec<f16>,
}

impl LutData {
    fn new(dimensions: [u32; 3], texels: Vec<f16>) -> Result<Self, AtmosphereError> {
        let expected = dimensions
            .into_iter()
            .try_fold(RGBA_CHANNELS as u64, |n, axis| {
                n.checked_mul(u64::from(axis))
            })
            .ok_or(AtmosphereError::DimensionOverflow)?;
        if texels.len() as u64 != expected {
            return Err(AtmosphereError::InvalidConfig(format!(
                "RGBA16F payload has {} components, expected {expected}",
                texels.len()
            )));
        }
        if texels.iter().any(|v| !v.is_finite()) {
            return Err(AtmosphereError::InvalidConfig(
                "LUT payload contains a non-finite half value".into(),
            ));
        }
        Ok(Self { dimensions, texels })
    }
    pub fn byte_size(&self) -> u64 {
        (self.texels.len() * 2) as u64
    }
    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }
    pub fn texels(&self) -> &[f16] {
        &self.texels
    }
    pub fn as_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.texels.len() * 2);
        for v in &self.texels {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes
    }
    pub fn rgba_f32(&self) -> Vec<f32> {
        self.texels.iter().map(|v| v.to_f32()).collect()
    }
    pub fn sample_trilinear(&self, uvw: [f32; 3]) -> [f32; 4] {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        let mut frac = [0.0; 3];
        for a in 0..3 {
            let n = self.dimensions[a] as usize;
            let p = uvw[a].clamp(0.0, 1.0) * (n - 1) as f32;
            lo[a] = p.floor() as usize;
            hi[a] = (lo[a] + 1).min(n - 1);
            frac[a] = p - lo[a] as f32;
        }
        let mut out = [0.0; 4];
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let w = (if x == 0 { 1.0 - frac[0] } else { frac[0] })
                        * (if y == 0 { 1.0 - frac[1] } else { frac[1] })
                        * (if z == 0 { 1.0 - frac[2] } else { frac[2] });
                    let v = self.fetch_rgba(
                        if x == 0 { lo[0] } else { hi[0] },
                        if y == 0 { lo[1] } else { hi[1] },
                        if z == 0 { lo[2] } else { hi[2] },
                    );
                    for c in 0..4 {
                        out[c] += w * v[c];
                    }
                }
            }
        }
        out
    }
    fn fetch_rgba(&self, x: usize, y: usize, z: usize) -> [f32; 4] {
        let i = ((z * self.dimensions[1] as usize + y) * self.dimensions[0] as usize + x) * 4;
        [
            self.texels[i].to_f32(),
            self.texels[i + 1].to_f32(),
            self.texels[i + 2].to_f32(),
            self.texels[i + 3].to_f32(),
        ]
    }
}

#[derive(Debug)]
pub struct AtmosphereLuts {
    pub(crate) metadata: AtmosphereLutMetadata,
    pub(crate) transmittance: LutData,
    pub(crate) single_scattering: LutData,
    pub(crate) multiple_scattering: LutData,
    pub(crate) aerial_perspective: LutData,
    /// Mean absolute total-transport contribution for each discrete order.
    /// This conservative convergence series includes the internal Lambertian
    /// boundary even though the public scattering payload stores volume light
    /// only.
    pub(crate) order_deltas: Vec<f32>,
    _sealed_deterministic_sha256: [u8; 32],
    _host_visible_allocation: ResourceHandle,
}

impl AtmosphereLuts {
    pub fn metadata(&self) -> &AtmosphereLutMetadata {
        &self.metadata
    }
    pub fn transmittance(&self) -> &LutData {
        &self.transmittance
    }
    pub fn single_scattering(&self) -> &LutData {
        &self.single_scattering
    }
    pub fn accumulated_scattering(&self) -> &LutData {
        &self.multiple_scattering
    }
    pub fn aerial_perspective(&self) -> &LutData {
        &self.aerial_perspective
    }
    pub fn order_deltas(&self) -> &[f32] {
        &self.order_deltas
    }
    pub fn byte_size(&self) -> u64 {
        self.transmittance.byte_size()
            + self.single_scattering.byte_size()
            + self.multiple_scattering.byte_size()
            + self.aerial_perspective.byte_size()
    }
    fn visit_deterministic_bytes(&self, mut visit: impl FnMut(&[u8])) {
        visit(b"AETHER-LUT-v2\0");
        for d in [
            self.metadata.dimensions.transmittance_mu,
            self.metadata.dimensions.transmittance_height,
            self.metadata.dimensions.scattering_mu_view,
            self.metadata.dimensions.scattering_mu_sun,
            self.metadata.dimensions.scattering_height,
            self.metadata.dimensions.scattering_nu,
            self.metadata.dimensions.aerial_distance,
            self.metadata.dimensions.aerial_mu_view,
            self.metadata.dimensions.aerial_height,
        ] {
            visit(&d.to_le_bytes());
        }
        for d in [
            self.transmittance.dimensions,
            self.single_scattering.dimensions,
            self.multiple_scattering.dimensions,
            self.aerial_perspective.dimensions,
        ]
        .into_iter()
        .flatten()
        {
            visit(&d.to_le_bytes());
        }
        for v in [
            self.metadata.config.turbidity,
            self.metadata.config.ozone_du,
            self.metadata.config.mie_g,
            self.metadata.config.bottom_radius_m,
            self.metadata.config.top_radius_m,
            self.metadata.config.rayleigh_scale_height_m,
            self.metadata.config.mie_scale_height_m,
            self.metadata.config.max_aerial_distance_m,
            self.metadata.config.ground_albedo,
        ] {
            visit(&v.to_bits().to_le_bytes());
        }
        visit(&self.metadata.config.scattering_orders.to_le_bytes());
        for wavelength in self.metadata.wavelengths_nm {
            visit(&wavelength.to_bits().to_le_bytes());
        }
        visit(&self.metadata.scattering_orders.to_le_bytes());
        for text in [
            self.metadata.scattering_lut_semantics,
            self.metadata.storage_format,
        ] {
            visit(&(text.len() as u64).to_le_bytes());
            visit(text.as_bytes());
        }
        let precomputed = [u8::from(self.metadata.precomputed)];
        visit(&precomputed);
        match self.metadata.precomputed_turbidity_bracket {
            Some(bracket) => {
                visit(&[1]);
                for value in bracket {
                    visit(&value.to_bits().to_le_bytes());
                }
            }
            None => visit(&[0]),
        }
        for lut in [
            &self.transmittance,
            &self.single_scattering,
            &self.multiple_scattering,
            &self.aerial_perspective,
        ] {
            for value in &lut.texels {
                visit(&value.to_bits().to_le_bytes());
            }
        }
        for d in &self.order_deltas {
            visit(&d.to_le_bytes());
        }
    }
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.byte_size() as usize + 256);
        self.visit_deterministic_bytes(|chunk| bytes.extend_from_slice(chunk));
        bytes
    }
    pub fn deterministic_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        self.visit_deterministic_bytes(|chunk| hasher.update(chunk));
        hasher.finalize().into()
    }
    fn seal(mut self) -> Self {
        self._sealed_deterministic_sha256 = self.deterministic_sha256();
        self
    }
    pub(crate) fn sealed_deterministic_sha256(&self) -> [u8; 32] {
        self._sealed_deterministic_sha256
    }
    pub fn sample_transmittance(&self, altitude_m: f32, mu: f32) -> [f32; 4] {
        let h = self.metadata.config.atmosphere_height_m();
        self.transmittance
            .sample_trilinear([(mu + 1.0) * 0.5, altitude_m.clamp(0.0, h) / h, 0.0])
    }
    pub fn sample_multiple_scattering(
        &self,
        altitude_m: f32,
        mu_sun: f32,
        mu_view: f32,
        nu: f32,
    ) -> [f32; 4] {
        let d = self.metadata.dimensions;
        let coordinates = [
            scattering_mu_to_unit(mu_view) * (d.scattering_mu_view - 1) as f32,
            scattering_mu_to_unit(mu_sun) * (d.scattering_mu_sun - 1) as f32,
            scattering_height_to_unit(altitude_m, self.metadata.config.atmosphere_height_m())
                * (d.scattering_height - 1) as f32,
            scattering_nu_to_unit(nu) * (d.scattering_nu - 1) as f32,
        ];
        let ext = [
            d.scattering_mu_view as usize,
            d.scattering_mu_sun as usize,
            d.scattering_height as usize,
            d.scattering_nu as usize,
        ];
        let lo = coordinates.map(|p| p.floor() as usize);
        let hi: [usize; 4] = std::array::from_fn(|a| (lo[a] + 1).min(ext[a] - 1));
        let f: [f32; 4] = std::array::from_fn(|a| coordinates[a] - lo[a] as f32);
        let mut out = [0.0; 4];
        for n in 0..2 {
            for h in 0..2 {
                for s in 0..2 {
                    for v in 0..2 {
                        let sides = [v, s, h, n];
                        let mut w = 1.0;
                        let mut i = [0usize; 4];
                        for a in 0..4 {
                            i[a] = if sides[a] == 0 { lo[a] } else { hi[a] };
                            w *= if sides[a] == 0 { 1.0 - f[a] } else { f[a] };
                        }
                        let value =
                            self.multiple_scattering
                                .fetch_rgba(i[0], i[1], i[2] * ext[3] + i[3]);
                        for c in 0..4 {
                            out[c] += w * value[c];
                        }
                    }
                }
            }
        }
        out
    }
    pub fn sample_aerial_perspective(
        &self,
        altitude_m: f32,
        mu_view: f32,
        distance_m: f32,
    ) -> [f32; 4] {
        let c = &self.metadata.config;
        self.aerial_perspective.sample_trilinear([
            distance_m.clamp(0.0, c.max_aerial_distance_m) / c.max_aerial_distance_m,
            (mu_view + 1.0) * 0.5,
            altitude_m.clamp(0.0, c.atmosphere_height_m()) / c.atmosphere_height_m(),
        ])
    }
    pub fn sky_radiance(
        &self,
        observer_altitude_m: f32,
        view_dir: [f32; 3],
        sun_dir: [f32; 3],
    ) -> Result<[f32; 3], AtmosphereError> {
        validate_observer(&self.metadata.config, observer_altitude_m)?;
        let view = normalized(view_dir)?;
        let sun = normalized(sun_dir)?;
        let s =
            self.sample_multiple_scattering(observer_altitude_m, sun[1], view[1], dot(view, sun));
        Ok([s[0], s[1], s[2]])
    }
    pub fn apply_aerial_perspective(
        &self,
        surface_rgb: [f32; 3],
        observer_altitude_m: f32,
        distance_m: f32,
        view_dir: [f32; 3],
        sun_dir: [f32; 3],
    ) -> Result<[f32; 3], AtmosphereError> {
        validate_observer(&self.metadata.config, observer_altitude_m)?;
        if surface_rgb.iter().any(|v| !v.is_finite()) || !distance_m.is_finite() || distance_m < 0.0
        {
            return Err(AtmosphereError::InvalidConfig(
                "surface RGB and aerial distance must be finite and nonnegative".into(),
            ));
        }
        let view = normalized(view_dir)?;
        let sun = normalized(sun_dir)?;
        let distance = distance_m.min(self.metadata.config.max_aerial_distance_m);
        let t = self.sample_aerial_perspective(observer_altitude_m, view[1], distance)[3]
            .clamp(0.0, 1.0);
        let tb = self.sample_transmittance(observer_altitude_m, view[1])[3].clamp(0.0, 1.0);
        let f = if 1.0 - tb > 1.0e-6 {
            ((1.0 - t) / (1.0 - tb)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let s =
            self.sample_multiple_scattering(observer_altitude_m, sun[1], view[1], dot(view, sun));
        Ok(std::array::from_fn(|c| {
            surface_rgb[c] * t + s[c].max(0.0) * f
        }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceEnvironment {
    pub width: u32,
    pub height: u32,
    pub rgb_linear: Vec<f32>,
}

fn tracker_handle(bytes: u64, label: &str) -> Result<ResourceHandle, AtmosphereError> {
    if bytes > HOST_VISIBLE_LIMIT_BYTES {
        return Err(AtmosphereError::BudgetExceeded {
            requested: bytes,
            limit: HOST_VISIBLE_LIMIT_BYTES,
        });
    }
    tracked_host_allocation(bytes, label)
        .map_err(|error| AtmosphereError::TrackerBudget(error.to_string()))
}

fn precomputed_bracket(t: f32) -> Result<(usize, usize, f32), AtmosphereError> {
    if !(TURBIDITY_BANK[0]..=TURBIDITY_BANK[4]).contains(&t) {
        return Err(AtmosphereError::UnsupportedPrecomputedConfig(format!(
            "turbidity {t}; shipped range is [1, 10]"
        )));
    }
    for i in 0..4 {
        let a = TURBIDITY_BANK[i];
        let b = TURBIDITY_BANK[i + 1];
        if t <= b {
            return Ok((i, i + 1, (t - a) / (b - a)));
        }
    }
    Ok((4, 4, 0.0))
}

pub fn load_precomputed_atmosphere_luts(
    config: AtmosphereConfig,
) -> Result<AtmosphereLuts, AtmosphereError> {
    config.validate()?;
    let defaults = AtmosphereConfig::default();
    if config.dimensions != LutDimensions::precomputed() {
        return Err(AtmosphereError::UnsupportedPrecomputedConfig(format!(
            "dimensions={:?}; shipped dimensions are {:?}",
            config.dimensions,
            LutDimensions::precomputed()
        )));
    }
    let exact = |a: f32, b: f32| a.to_bits() == b.to_bits();
    let fixed = [
        (config.ozone_du, defaults.ozone_du, "ozone_du"),
        (config.mie_g, defaults.mie_g, "mie_g"),
        (
            config.bottom_radius_m,
            defaults.bottom_radius_m,
            "bottom_radius_m",
        ),
        (config.top_radius_m, defaults.top_radius_m, "top_radius_m"),
        (
            config.rayleigh_scale_height_m,
            defaults.rayleigh_scale_height_m,
            "rayleigh_scale_height_m",
        ),
        (
            config.mie_scale_height_m,
            defaults.mie_scale_height_m,
            "mie_scale_height_m",
        ),
        (
            config.max_aerial_distance_m,
            defaults.max_aerial_distance_m,
            "max_aerial_distance_m",
        ),
        (
            config.ground_albedo,
            defaults.ground_albedo,
            "ground_albedo",
        ),
    ];
    if let Some((a, b, n)) = fixed.into_iter().find(|(a, b, _)| !exact(*a, *b)) {
        return Err(AtmosphereError::UnsupportedPrecomputedConfig(format!(
            "{n}={a}; shipped value is {b}"
        )));
    }
    if config.scattering_orders != 4 {
        return Err(AtmosphereError::UnsupportedPrecomputedConfig(format!(
            "scattering_orders={}; shipped value is 4",
            config.scattering_orders
        )));
    }
    let (lower, upper, factor) = precomputed_bracket(config.turbidity)?;
    let handle = tracker_handle(config.dimensions.payload_bytes()?, "aether-precomputed-lut")?;
    let p = precomputed::interpolate(lower, upper, factor);
    Ok(AtmosphereLuts {
        metadata: AtmosphereLutMetadata {
            dimensions: config.dimensions,
            wavelengths_nm: WAVELENGTHS_NM,
            scattering_orders: 4,
            scattering_lut_semantics: ACCUMULATED_SCATTERING_LUT_SEMANTICS,
            storage_format: "rgba16float",
            precomputed: true,
            precomputed_turbidity_bracket: Some([TURBIDITY_BANK[lower], TURBIDITY_BANK[upper]]),
            config,
        },
        transmittance: LutData::new(precomputed::TRANSMITTANCE_DIMENSIONS, p.transmittance)?,
        single_scattering: LutData::new(precomputed::SCATTERING_DIMENSIONS, p.single_scattering)?,
        multiple_scattering: LutData::new(
            precomputed::SCATTERING_DIMENSIONS,
            p.accumulated_scattering,
        )?,
        aerial_perspective: LutData::new(precomputed::AERIAL_DIMENSIONS, p.aerial_perspective)?,
        order_deltas: p.order_deltas,
        _sealed_deterministic_sha256: [0; 32],
        _host_visible_allocation: handle,
    }
    .seal())
}
pub fn default_atmosphere_luts() -> AtmosphereLuts {
    load_precomputed_atmosphere_luts(AtmosphereConfig::default())
        .expect("the shipped default atmosphere bank must be valid")
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn normalized(v: [f32; 3]) -> Result<[f32; 3], AtmosphereError> {
    if v.iter().any(|x| !x.is_finite()) || dot(v, v) <= 1.0e-12 {
        return Err(AtmosphereError::InvalidConfig(
            "direction vectors must be finite and non-zero".into(),
        ));
    }
    let r = dot(v, v).sqrt().recip();
    Ok([v[0] * r, v[1] * r, v[2] * r])
}
fn density_at(c: &AtmosphereConfig, h: f32) -> [f32; 3] {
    let h = h.max(0.0);
    [
        (-h / c.rayleigh_scale_height_m).exp(),
        (-h / c.mie_scale_height_m).exp(),
        (1.0 - ((h - 25_000.0) / 15_000.0).abs()).max(0.0) * c.ozone_du / 300.0,
    ]
}
fn ozone_absorption(w: f32) -> f32 {
    1.2e-6 * (-0.5 * ((w - 600.0) / 85.0).powi(2)).exp()
}
fn distance_to_top(c: &AtmosphereConfig, h: f32, mu: f32) -> f32 {
    let r = c.bottom_radius_m + h.clamp(0.0, c.atmosphere_height_m());
    let radial = r * mu;
    let discriminant = radial * radial + (c.top_radius_m - r) * (c.top_radius_m + r);
    (-radial + discriminant.max(0.0).sqrt()).max(0.0)
}
fn distance_to_ground(c: &AtmosphereConfig, h: f32, mu: f32) -> Option<f32> {
    if mu >= 0.0 {
        return None;
    }
    let r = c.bottom_radius_m + h.clamp(0.0, c.atmosphere_height_m());
    let radial = r * mu;
    let d = radial * radial - (r - c.bottom_radius_m) * (r + c.bottom_radius_m);
    if d < 0.0 {
        None
    } else {
        let s = -radial - d.sqrt();
        (s >= 0.0).then_some(s)
    }
}
fn distance_to_boundary(c: &AtmosphereConfig, h: f32, mu: f32) -> f32 {
    distance_to_ground(c, h, mu).unwrap_or_else(|| distance_to_top(c, h, mu))
}
fn altitude_along(c: &AtmosphereConfig, h: f32, mu: f32, s: f32) -> f32 {
    let r = c.bottom_radius_m + h.clamp(0.0, c.atmosphere_height_m());
    (r * r + s * s + 2.0 * r * mu * s).max(0.0).sqrt() - c.bottom_radius_m
}
fn optical_columns(c: &AtmosphereConfig, h: f32, mu: f32, d: f32, steps: usize) -> [f32; 3] {
    if d <= 0.0 {
        return [0.0; 3];
    }
    let ds = d / steps as f32;
    let mut out = [0.0; 3];
    for i in 0..steps {
        let rho = density_at(c, altitude_along(c, h, mu, (i as f32 + 0.5) * ds));
        for k in 0..3 {
            out[k] += rho[k] * ds;
        }
    }
    out
}
fn transmittance_from_columns(c: &AtmosphereConfig, col: [f32; 3]) -> [f32; NUM_WAVELENGTHS] {
    let mie = c.mie();
    std::array::from_fn(|i| {
        let w = WAVELENGTHS_NM[i];
        (-(rayleigh_scattering_coefficient(w) * col[0]
            + mie.extinction(w) * col[1]
            + ozone_absorption(w) * col[2])
            .max(0.0))
        .exp()
    })
}

fn extinction_at_density(c: &AtmosphereConfig, density: [f32; 3], wavelength_nm: f32) -> f32 {
    let mie = c.mie();
    (rayleigh_scattering_coefficient(wavelength_nm) * density[0]
        + mie.extinction(wavelength_nm) * density[1]
        + ozone_absorption(wavelength_nm) * density[2])
        .max(0.0)
}

/// Exact path length under piecewise-constant extinction within one cell.
/// This avoids losing the near endpoint of an optically thick cell, while the
/// small-extinction branch preserves the analytic `ds` limit.
fn attenuated_cell_length(extinction: f32, ds: f32) -> f32 {
    if extinction <= 1.0e-12 {
        ds
    } else {
        -(-extinction * ds).exp_m1() / extinction
    }
}

fn transmittance_segment(c: &AtmosphereConfig, h: f32, mu: f32, d: f32) -> [f32; NUM_WAVELENGTHS] {
    transmittance_from_columns(c, optical_columns(c, h, mu, d, 64))
}

fn integrate_single_scattering(
    c: &AtmosphereConfig,
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    nu: f32,
    distance_limit: f32,
    steps: usize,
) -> ([f32; NUM_WAVELENGTHS], [f32; NUM_WAVELENGTHS]) {
    let length = distance_to_boundary(c, h, mu_view).min(distance_limit);
    if length <= 0.0 {
        return ([0.0; NUM_WAVELENGTHS], [1.0; NUM_WAVELENGTHS]);
    }
    let ds = length / steps as f32;
    let mie = c.mie();
    let mut radiance = [0.0; NUM_WAVELENGTHS];
    let mut view_columns = [0.0; 3];
    for i in 0..steps {
        let s = (i as f32 + 0.5) * ds;
        let (sample_h, local_mu_sun) =
            ray_sample_altitude_and_sun_cosine(c, h, mu_view, mu_sun, nu, s);
        let rho = density_at(c, sample_h);
        let view_start = transmittance_from_columns(c, view_columns);
        if distance_to_ground(c, sample_h, local_mu_sun).is_none() {
            let sun_distance = distance_to_top(c, sample_h, local_mu_sun);
            let sun_columns = optical_columns(c, sample_h, local_mu_sun, sun_distance, 64);
            let sun_transmittance = transmittance_from_columns(c, sun_columns);
            for w in 0..NUM_WAVELENGTHS {
                let wl = WAVELENGTHS_NM[w];
                let scatter = rayleigh_scattering_coefficient(wl) * rho[0] * rayleigh_phase(nu)
                    + mie.scattering(wl) * rho[1] * mie_phase_cornette_shanks(nu, mie.g);
                let cell_length = attenuated_cell_length(extinction_at_density(c, rho, wl), ds);
                radiance[w] += view_start[w] * sun_transmittance[w] * scatter * cell_length;
            }
        }
        for k in 0..3 {
            view_columns[k] += rho[k] * ds;
        }
    }
    (radiance, transmittance_segment(c, h, mu_view, length))
}

fn integrate_ground_bounce_scattering(
    c: &AtmosphereConfig,
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    distance_limit: f32,
    steps: usize,
) -> [f32; NUM_WAVELENGTHS] {
    if c.ground_albedo <= 0.0 || mu_sun <= 0.0 {
        return [0.0; NUM_WAVELENGTHS];
    }
    let length = distance_to_boundary(c, h, mu_view).min(distance_limit);
    if length <= 0.0 {
        return [0.0; NUM_WAVELENGTHS];
    }
    let ground_t = transmittance_segment(c, 0.0, mu_sun, distance_to_top(c, 0.0, mu_sun));
    let ground: [f32; NUM_WAVELENGTHS] =
        std::array::from_fn(|w| c.ground_albedo * mu_sun * ground_t[w] / std::f32::consts::PI);
    let ds = length / steps as f32;
    let mie = c.mie();
    let cosine = (-mu_view).clamp(-1.0, 1.0);
    let mut radiance = [0.0; NUM_WAVELENGTHS];
    let mut view_columns = [0.0; 3];
    for i in 0..steps {
        let s = (i as f32 + 0.5) * ds;
        let sample_h = altitude_along(c, h, mu_view, s);
        let rho = density_at(c, sample_h);
        let view_start = transmittance_from_columns(c, view_columns);
        let tg = transmittance_from_columns(c, optical_columns(c, 0.0, 1.0, sample_h.max(0.0), 64));
        for w in 0..NUM_WAVELENGTHS {
            let wl = WAVELENGTHS_NM[w];
            let scatter = rayleigh_scattering_coefficient(wl) * rho[0] * rayleigh_phase(cosine)
                + mie.scattering(wl) * rho[1] * mie_phase_cornette_shanks(cosine, mie.g);
            let cell_length = attenuated_cell_length(extinction_at_density(c, rho, wl), ds);
            radiance[w] += view_start[w] * tg[w] * ground[w] * scatter * cell_length;
        }
        for k in 0..3 {
            view_columns[k] += rho[k] * ds;
        }
    }
    radiance
}

fn validate_observer(c: &AtmosphereConfig, h: f32) -> Result<(), AtmosphereError> {
    if !h.is_finite() || !(0.0..=c.atmosphere_height_m()).contains(&h) {
        Err(AtmosphereError::InvalidConfig(format!(
            "observer altitude must be in [0, {}] metres",
            c.atmosphere_height_m()
        )))
    } else {
        Ok(())
    }
}

pub fn reference_sky_radiance(
    c: &AtmosphereConfig,
    h: f32,
    view_dir: [f32; 3],
    sun_dir: [f32; 3],
) -> Result<[f32; 3], AtmosphereError> {
    c.validate()?;
    validate_observer(c, h)?;
    let v = normalized(view_dir)?;
    let s = normalized(sun_dir)?;
    let (single, _) = integrate_single_scattering(c, h, v[1], s[1], dot(v, s), f32::MAX, 64);
    let ground = integrate_ground_bounce_scattering(c, h, v[1], s[1], f32::MAX, 64);
    let spectrum: [f32; NUM_WAVELENGTHS] = std::array::from_fn(|w| single[w] + ground[w]);
    Ok(spectral_to_linear_rgb(&spectrum).map(|x| x.max(0.0)))
}

pub fn reference_aerial_radiance(
    c: &AtmosphereConfig,
    surface: [f32; 3],
    h: f32,
    distance: f32,
    view_dir: [f32; 3],
    sun_dir: [f32; 3],
) -> Result<[f32; 3], AtmosphereError> {
    c.validate()?;
    validate_observer(c, h)?;
    if surface.iter().any(|v| !v.is_finite()) || !distance.is_finite() || distance < 0.0 {
        return Err(AtmosphereError::InvalidConfig(
            "surface RGB and aerial distance must be finite and nonnegative".into(),
        ));
    }
    let v = normalized(view_dir)?;
    let s = normalized(sun_dir)?;
    let d = distance.min(c.max_aerial_distance_m);
    let (single, t) = integrate_single_scattering(c, h, v[1], s[1], dot(v, s), d, 64);
    let ground = integrate_ground_bounce_scattering(c, h, v[1], s[1], d, 64);
    let scatter: [f32; NUM_WAVELENGTHS] = std::array::from_fn(|w| single[w] + ground[w]);
    let rgb = spectral_to_linear_rgb(&scatter);
    let tr = spectral_to_linear_rgb(&t);
    Ok(std::array::from_fn(|k| {
        (surface[k] * tr[k].clamp(0.0, 1.0) + rgb[k].max(0.0)).max(0.0)
    }))
}

pub fn generate_reference_equirectangular(
    c: &AtmosphereConfig,
    width: u32,
    height: u32,
    h: f32,
    sun_dir: [f32; 3],
) -> Result<ReferenceEnvironment, AtmosphereError> {
    c.validate()?;
    validate_observer(c, h)?;
    let count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(AtmosphereError::DimensionOverflow)?;
    let bytes = count
        .checked_mul(12)
        .ok_or(AtmosphereError::DimensionOverflow)?;
    if width < 2 || height < 2 || width > 2048 || height > 1024 || bytes > HOST_VISIBLE_LIMIT_BYTES
    {
        return Err(AtmosphereError::InvalidConfig(
            "reference environment must be 2..2048 by 2..1024 and fit the host budget".into(),
        ));
    }
    let sun = normalized(sun_dir)?;
    let mut rgb = Vec::with_capacity(count as usize * 3);
    for y in 0..height {
        let lat =
            std::f32::consts::FRAC_PI_2 - std::f32::consts::PI * (y as f32 + 0.5) / height as f32;
        let cl = lat.cos();
        for x in 0..width {
            let lon =
                2.0 * std::f32::consts::PI * (x as f32 + 0.5) / width as f32 - std::f32::consts::PI;
            rgb.extend_from_slice(&reference_sky_radiance(
                c,
                h,
                [cl * lon.sin(), lat.sin(), cl * lon.cos()],
                sun,
            )?);
        }
    }
    Ok(ReferenceEnvironment {
        width,
        height,
        rgb_linear: rgb,
    })
}

#[cfg(feature = "atmosphere-bake")]
#[derive(Clone, Copy, Debug)]
struct RaySampleGeometry {
    altitude_m: f32,
    mu_sun: f32,
    outgoing: [f32; 3],
    sun: [f32; 3],
    up: [f32; 3],
    tangent: [f32; 3],
}

fn ray_sample_vectors(
    c: &AtmosphereConfig,
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    nu: f32,
    distance: f32,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3], f32) {
    let mv = mu_view.clamp(-1.0, 1.0);
    let ms = mu_sun.clamp(-1.0, 1.0);
    let vx = (1.0 - mv * mv).max(0.0).sqrt();
    let sh = (1.0 - ms * ms).max(0.0).sqrt();
    let requested = if vx > 1.0e-6 {
        (nu.clamp(-1.0, 1.0) - mv * ms) / vx
    } else {
        0.0
    };
    let sx = requested.clamp(-sh, sh);
    let sz = (sh * sh - sx * sx).max(0.0).sqrt();
    let outgoing = [vx, mv, 0.0];
    let sun = [sx, ms, sz];
    let r = c.bottom_radius_m + h.clamp(0.0, c.atmosphere_height_m());
    let position = [outgoing[0] * distance, r + outgoing[1] * distance, 0.0];
    let sr = dot(position, position).sqrt().max(c.bottom_radius_m);
    let up = [position[0] / sr, position[1] / sr, 0.0];
    (outgoing, sun, position, up, sr - c.bottom_radius_m)
}

fn ray_sample_altitude_and_sun_cosine(
    c: &AtmosphereConfig,
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    nu: f32,
    distance: f32,
) -> (f32, f32) {
    let (_, sun, _, up, altitude_m) = ray_sample_vectors(c, h, mu_view, mu_sun, nu, distance);
    (
        altitude_m.clamp(0.0, c.atmosphere_height_m()),
        dot(sun, up).clamp(-1.0, 1.0),
    )
}

#[cfg(feature = "atmosphere-bake")]
fn ray_sample_geometry(
    c: &AtmosphereConfig,
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    nu: f32,
    distance: f32,
) -> RaySampleGeometry {
    let (outgoing, sun, _, up, altitude_m) =
        ray_sample_vectors(c, h, mu_view, mu_sun, nu, distance);
    let tangent = [up[1], -up[0], 0.0];
    RaySampleGeometry {
        altitude_m: altitude_m.clamp(0.0, c.atmosphere_height_m()),
        mu_sun: dot(sun, up).clamp(-1.0, 1.0),
        outgoing,
        sun,
        up,
        tangent,
    }
}

#[cfg(feature = "atmosphere-bake")]
const ANGULAR_GAUSS_LEGENDRE: [(f32, f32, f32); 16] = [
    (-0.989_400_9, 0.145_209_48, 0.027_152_46),
    (-0.944_575, 0.328_295_65, 0.062_253_524),
    (-0.865_631_2, 0.500_682_2, 0.095_158_51),
    (-0.755_404_4, 0.655_258_9, 0.124_628_97),
    (-0.617_876_23, 0.786_275_4, 0.149_595_99),
    (-0.458_016_78, 0.888_943_55, 0.169_156_52),
    (-0.281_603_54, 0.959_530_83, 0.182_603_42),
    (-0.095_012_51, 0.995_476_07, 0.189_450_6),
    (0.095_012_51, 0.995_476_07, 0.189_450_6),
    (0.281_603_54, 0.959_530_83, 0.182_603_42),
    (0.458_016_78, 0.888_943_55, 0.169_156_52),
    (0.617_876_23, 0.786_275_4, 0.149_595_99),
    (0.755_404_4, 0.655_258_9, 0.124_628_97),
    (0.865_631_2, 0.500_682_2, 0.095_158_51),
    (0.944_575, 0.328_295_65, 0.062_253_524),
    (0.989_400_9, 0.145_209_48, 0.027_152_46),
];

#[cfg(feature = "atmosphere-bake")]
const ANGULAR_AZIMUTHS: [[f32; 2]; 32] = [
    [0.995_184_7, 0.098_017_14],
    [0.956_940_35, 0.290_284_66],
    [0.881_921_3, 0.471_396_74],
    [0.773_010_43, 0.634_393_3],
    [0.634_393_3, 0.773_010_43],
    [0.471_396_74, 0.881_921_3],
    [0.290_284_66, 0.956_940_35],
    [0.098_017_14, 0.995_184_7],
    [-0.098_017_14, 0.995_184_7],
    [-0.290_284_66, 0.956_940_35],
    [-0.471_396_74, 0.881_921_3],
    [-0.634_393_3, 0.773_010_43],
    [-0.773_010_43, 0.634_393_3],
    [-0.881_921_3, 0.471_396_74],
    [-0.956_940_35, 0.290_284_66],
    [-0.995_184_7, 0.098_017_14],
    [-0.995_184_7, -0.098_017_14],
    [-0.956_940_35, -0.290_284_66],
    [-0.881_921_3, -0.471_396_74],
    [-0.773_010_43, -0.634_393_3],
    [-0.634_393_3, -0.773_010_43],
    [-0.471_396_74, -0.881_921_3],
    [-0.290_284_66, -0.956_940_35],
    [-0.098_017_14, -0.995_184_7],
    [0.098_017_14, -0.995_184_7],
    [0.290_284_66, -0.956_940_35],
    [0.471_396_74, -0.881_921_3],
    [0.634_393_3, -0.773_010_43],
    [0.773_010_43, -0.634_393_3],
    [0.881_921_3, -0.471_396_74],
    [0.956_940_35, -0.290_284_66],
    [0.995_184_7, -0.098_017_14],
];

/// Positive-weight 16x32 tensor-product sphere rule.  The previous octahedral
/// rule sampled only |mu|=0.577, so the bright blue near-horizon field was
/// absent from every higher-order gather. These dense offline ordinates
/// resolve the horizon and keep the raw g=0.8 Mie phase integral within one
/// percent across representative outgoing directions without changing LUT
/// axes.
#[cfg(feature = "atmosphere-bake")]
fn angular_quadrature() -> &'static [([f32; 3], f32)] {
    static QUADRATURE: std::sync::OnceLock<Vec<([f32; 3], f32)>> = std::sync::OnceLock::new();
    QUADRATURE.get_or_init(|| {
        let azimuth_weight = std::f32::consts::TAU / ANGULAR_AZIMUTHS.len() as f32;
        ANGULAR_GAUSS_LEGENDRE
            .into_iter()
            .flat_map(|(mu, radial, mu_weight)| {
                ANGULAR_AZIMUTHS.into_iter().map(move |azimuth| {
                    (
                        [radial * azimuth[0], mu, radial * azimuth[1]],
                        mu_weight * azimuth_weight,
                    )
                })
            })
            .collect()
    })
}

#[cfg(feature = "atmosphere-bake")]
fn scattering_index(d: LutDimensions, h: usize, n: usize, s: usize, v: usize) -> usize {
    (((h * d.scattering_nu as usize + n) * d.scattering_mu_sun as usize + s)
        * d.scattering_mu_view as usize)
        + v
}

#[cfg(feature = "atmosphere-bake")]
fn sample_spectral_scattering(
    values: &[[f32; NUM_WAVELENGTHS]],
    d: LutDimensions,
    atmosphere_height: f32,
    h: f32,
    mu_sun: f32,
    mu_view: f32,
    nu: f32,
) -> [f32; NUM_WAVELENGTHS] {
    let p = [
        scattering_height_to_unit(h, atmosphere_height) * (d.scattering_height - 1) as f32,
        scattering_nu_to_unit(nu) * (d.scattering_nu - 1) as f32,
        scattering_mu_to_unit(mu_sun) * (d.scattering_mu_sun - 1) as f32,
        scattering_mu_to_unit(mu_view) * (d.scattering_mu_view - 1) as f32,
    ];
    let ext = [
        d.scattering_height as usize,
        d.scattering_nu as usize,
        d.scattering_mu_sun as usize,
        d.scattering_mu_view as usize,
    ];
    let lo = p.map(|x| x.floor() as usize);
    let hi: [usize; 4] = std::array::from_fn(|a| (lo[a] + 1).min(ext[a] - 1));
    let f: [f32; 4] = std::array::from_fn(|a| p[a] - lo[a] as f32);
    let mut out = [0.0; NUM_WAVELENGTHS];
    for hs in 0..2 {
        for ns in 0..2 {
            for ss in 0..2 {
                for vs in 0..2 {
                    let sides = [hs, ns, ss, vs];
                    let mut w = 1.0;
                    let mut i = [0usize; 4];
                    for a in 0..4 {
                        i[a] = if sides[a] == 0 { lo[a] } else { hi[a] };
                        w *= if sides[a] == 0 { 1.0 - f[a] } else { f[a] };
                    }
                    let q = values[scattering_index(d, i[0], i[1], i[2], i[3])];
                    for k in 0..NUM_WAVELENGTHS {
                        out[k] += w * q[k];
                    }
                }
            }
        }
    }
    out
}

#[cfg(feature = "atmosphere-bake")]
fn quadrature_direction(g: RaySampleGeometry, local: [f32; 3]) -> [f32; 3] {
    [
        g.tangent[0] * local[0] + g.up[0] * local[1],
        g.tangent[1] * local[0] + g.up[1] * local[1],
        local[2],
    ]
}

/// Normalize each discrete phase kernel against the directions actually used
/// by the gather. The positive-weight sphere rule is already close to unity;
/// normalization keeps a constant radiance field exactly energy preserving
/// for every outgoing direction and supported Mie asymmetry.
#[cfg(feature = "atmosphere-bake")]
fn phase_quadrature_normalization(g: RaySampleGeometry, mie_g: f32) -> [f32; 2] {
    let mut normalization = [0.0_f32; 2];
    for &(local, omega) in angular_quadrature() {
        let incoming = quadrature_direction(g, local);
        let cosine = dot(incoming, g.outgoing);
        normalization[0] += rayleigh_phase(cosine) * omega;
        normalization[1] += mie_phase_cornette_shanks(cosine, mie_g) * omega;
    }
    // Both phase functions are strictly positive for the supported g range.
    // Keep the clamp as a finite fail-safe for future quadrature edits.
    normalization.map(|value| value.max(1.0e-8))
}

#[cfg(feature = "atmosphere-bake")]
fn ground_boundary_source(
    c: &AtmosphereConfig,
    d: LutDimensions,
    incident: Option<&[[f32; NUM_WAVELENGTHS]]>,
    sun: [f32; 3],
    include_direct_sun: bool,
) -> [f32; NUM_WAVELENGTHS] {
    if c.ground_albedo <= 0.0 {
        return [0.0; NUM_WAVELENGTHS];
    }
    let mut irradiance = [0.0; NUM_WAVELENGTHS];
    if let Some(incident) = incident {
        let cosine_weight_sum = angular_quadrature()
            .iter()
            .filter(|(local, _)| local[1] > 0.0)
            .map(|(local, omega)| local[1] * omega)
            .sum::<f32>();
        let cosine_weight_normalization = std::f32::consts::PI / cosine_weight_sum;
        for &(local, omega) in angular_quadrature() {
            if local[1] <= 0.0 {
                continue;
            }
            let sample = sample_spectral_scattering(
                incident,
                d,
                c.atmosphere_height_m(),
                0.0,
                sun[1].clamp(-1.0, 1.0),
                local[1],
                dot(local, sun),
            );
            for w in 0..NUM_WAVELENGTHS {
                irradiance[w] += sample[w] * local[1] * omega * cosine_weight_normalization;
            }
        }
    }
    let mut radiance = irradiance.map(|v| v * c.ground_albedo / std::f32::consts::PI);
    // The order-2 boundary also receives the unscattered solar beam. Later
    // orders must not add it again: they receive its propagated contribution
    // through `previous`, just like every other order-n source term.
    if include_direct_sun && sun[1] > 0.0 {
        let mu_sun = sun[1].clamp(0.0, 1.0);
        let direct_transmittance =
            transmittance_segment(c, 0.0, mu_sun, distance_to_top(c, 0.0, mu_sun));
        for w in 0..NUM_WAVELENGTHS {
            radiance[w] +=
                c.ground_albedo * mu_sun * direct_transmittance[w] / std::f32::consts::PI;
        }
    }
    radiance
}

#[cfg(feature = "atmosphere-bake")]
#[allow(clippy::too_many_arguments)]
fn ground_boundary_along_ray(
    c: &AtmosphereConfig,
    d: LutDimensions,
    incident: Option<&[[f32; NUM_WAVELENGTHS]]>,
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    nu: f32,
    include_direct_sun: bool,
) -> [f32; NUM_WAVELENGTHS] {
    let Some(length) = distance_to_ground(c, h, mu_view) else {
        return [0.0; NUM_WAVELENGTHS];
    };
    let end = ray_sample_geometry(c, h, mu_view, mu_sun, nu, length);
    let sun_local = [dot(end.sun, end.tangent), dot(end.sun, end.up), end.sun[2]];
    let boundary = ground_boundary_source(c, d, incident, sun_local, include_direct_sun);
    let transmittance = transmittance_segment(c, h, mu_view, length);
    std::array::from_fn(|w| transmittance[w] * boundary[w])
}

#[cfg(feature = "atmosphere-bake")]
struct ScatteringOrderSample {
    /// Atmospheric volume contribution stored in the public scattering LUT.
    volume: [f32; NUM_WAVELENGTHS],
    /// Total discrete-ordinates field propagated into the next order. This is
    /// the volume term plus any attenuated Lambertian ground endpoint.
    transport: [f32; NUM_WAVELENGTHS],
}

#[cfg(feature = "atmosphere-bake")]
const SCATTERING_ORDER_STEPS: usize = 16;

#[cfg(feature = "atmosphere-bake")]
fn scattering_order_cell_edge(length: f32, index: usize, ground_bound: bool) -> f32 {
    let unit = index as f32 / SCATTERING_ORDER_STEPS as f32;
    if ground_bound {
        // Reverse quadratic: resolve the dense layer at the ground endpoint.
        length * (1.0 - (1.0 - unit) * (1.0 - unit))
    } else {
        // Forward quadratic: resolve the dense layer at the camera endpoint.
        length * unit * unit
    }
}

#[cfg(feature = "atmosphere-bake")]
#[allow(clippy::too_many_arguments)]
fn integrate_scattering_order(
    c: &AtmosphereConfig,
    d: LutDimensions,
    previous: &[[f32; NUM_WAVELENGTHS]],
    h: f32,
    mu_view: f32,
    mu_sun: f32,
    nu: f32,
) -> ScatteringOrderSample {
    let length = distance_to_boundary(c, h, mu_view);
    let mut volume = [0.0; NUM_WAVELENGTHS];
    if length > 0.0 {
        let ground_bound = distance_to_ground(c, h, mu_view).is_some();
        let mie = c.mie();
        let mut columns = [0.0; 3];
        for i in 0..SCATTERING_ORDER_STEPS {
            let start = scattering_order_cell_edge(length, i, ground_bound);
            let end = scattering_order_cell_edge(length, i + 1, ground_bound);
            let ds = end - start;
            let distance = 0.5 * (start + end);
            let g = ray_sample_geometry(c, h, mu_view, mu_sun, nu, distance);
            let rho = density_at(c, g.altitude_m);
            let view_start = transmittance_from_columns(c, columns);
            let phase_normalization = phase_quadrature_normalization(g, mie.g);
            let mut angular_source = [0.0; NUM_WAVELENGTHS];
            for &(local, omega) in angular_quadrature() {
                let incoming = quadrature_direction(g, local);
                let l = sample_spectral_scattering(
                    previous,
                    d,
                    c.atmosphere_height_m(),
                    g.altitude_m,
                    g.mu_sun,
                    dot(incoming, g.up),
                    dot(incoming, g.sun),
                );
                let cosine = dot(incoming, g.outgoing);
                for w in 0..NUM_WAVELENGTHS {
                    let wl = WAVELENGTHS_NM[w];
                    let scatter = rayleigh_scattering_coefficient(wl)
                        * rho[0]
                        * rayleigh_phase(cosine)
                        / phase_normalization[0]
                        + mie.scattering(wl) * rho[1] * mie_phase_cornette_shanks(cosine, mie.g)
                            / phase_normalization[1];
                    angular_source[w] += scatter * l[w] * omega;
                }
            }
            for w in 0..NUM_WAVELENGTHS {
                let cell_length =
                    attenuated_cell_length(extinction_at_density(c, rho, WAVELENGTHS_NM[w]), ds);
                volume[w] += view_start[w] * angular_source[w] * cell_length;
            }
            for k in 0..3 {
                columns[k] += rho[k] * ds;
            }
        }
    }
    let boundary = ground_boundary_along_ray(c, d, Some(previous), h, mu_view, mu_sun, nu, false);
    ScatteringOrderSample {
        volume,
        transport: std::array::from_fn(|w| volume[w] + boundary[w]),
    }
}

#[cfg(feature = "atmosphere-bake")]
fn rgba_from_spectral(s: &[f32; NUM_WAVELENGTHS], alpha: f32) -> [f16; 4] {
    let rgb = spectral_to_linear_rgb(s);
    [
        f16::from_f32(rgb[0].clamp(0.0, f16::MAX.to_f32())),
        f16::from_f32(rgb[1].clamp(0.0, f16::MAX.to_f32())),
        f16::from_f32(rgb[2].clamp(0.0, f16::MAX.to_f32())),
        f16::from_f32(alpha.clamp(0.0, f16::MAX.to_f32())),
    ]
}
#[cfg(feature = "atmosphere-bake")]
fn append_rgba(out: &mut Vec<f16>, rgba: [f16; 4]) {
    out.extend_from_slice(&rgba);
}

#[cfg(feature = "atmosphere-bake")]
pub fn bake_atmosphere_luts(config: AtmosphereConfig) -> Result<AtmosphereLuts, AtmosphereError> {
    config.validate()?;
    let d = config.dimensions;
    let payload_bytes = d.payload_bytes()?;
    let scatter_count_u64 = u64::from(d.scattering_mu_view)
        * u64::from(d.scattering_mu_sun)
        * u64::from(d.scattering_height)
        * u64::from(d.scattering_nu);
    let scratch = scatter_count_u64
        .checked_mul(NUM_WAVELENGTHS as u64)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_mul(4))
        .ok_or(AtmosphereError::DimensionOverflow)?;
    let combined = payload_bytes
        .checked_add(scratch)
        .ok_or(AtmosphereError::DimensionOverflow)?;
    if combined > HOST_VISIBLE_LIMIT_BYTES {
        return Err(AtmosphereError::BudgetExceeded {
            requested: combined,
            limit: HOST_VISIBLE_LIMIT_BYTES,
        });
    }
    let _scratch = tracker_handle(scratch, "aether-bake-spectral-scratch")?;
    let allocation = tracker_handle(payload_bytes, "aether-bake-lut-payload")?;
    let mut trans =
        Vec::with_capacity(d.transmittance_mu as usize * d.transmittance_height as usize * 4);
    for hi in 0..d.transmittance_height {
        let h = config.atmosphere_height_m() * hi as f32 / (d.transmittance_height - 1) as f32;
        for mi in 0..d.transmittance_mu {
            let mu = -1.0 + 2.0 * mi as f32 / (d.transmittance_mu - 1) as f32;
            let s = transmittance_segment(&config, h, mu, distance_to_boundary(&config, h, mu));
            append_rgba(
                &mut trans,
                rgba_from_spectral(&s, s.iter().sum::<f32>() / NUM_WAVELENGTHS as f32),
            );
        }
    }
    let count = scatter_count_u64 as usize;
    let mut single = Vec::with_capacity(count);
    let mut previous = Vec::with_capacity(count);
    for hi in 0..d.scattering_height {
        let h = scattering_height_from_unit(
            hi as f32 / (d.scattering_height - 1) as f32,
            config.atmosphere_height_m(),
        );
        for ni in 0..d.scattering_nu {
            let nu = scattering_nu_from_unit(ni as f32 / (d.scattering_nu - 1) as f32);
            for si in 0..d.scattering_mu_sun {
                let ms = scattering_mu_from_unit(si as f32 / (d.scattering_mu_sun - 1) as f32);
                for vi in 0..d.scattering_mu_view {
                    let mv = scattering_mu_from_unit(vi as f32 / (d.scattering_mu_view - 1) as f32);
                    let volume =
                        integrate_single_scattering(&config, h, mv, ms, nu, f32::MAX, 64).0;
                    let direct_ground =
                        ground_boundary_along_ray(&config, d, None, h, mv, ms, nu, true);
                    single.push(volume);
                    previous.push(std::array::from_fn(|w| volume[w] + direct_ground[w]));
                }
            }
        }
    }
    let mut accumulated = single.clone();
    let first = previous
        .iter()
        .flat_map(|sample| sample.iter())
        .map(|value| value.abs())
        .sum::<f32>()
        / (count * NUM_WAVELENGTHS) as f32;
    let mut deltas = vec![first];
    for _order in 2..=config.scattering_orders {
        let mut next = vec![[0.0; NUM_WAVELENGTHS]; count];
        for hi in 0..d.scattering_height as usize {
            let h = scattering_height_from_unit(
                hi as f32 / (d.scattering_height - 1) as f32,
                config.atmosphere_height_m(),
            );
            for ni in 0..d.scattering_nu as usize {
                let nu = scattering_nu_from_unit(ni as f32 / (d.scattering_nu - 1) as f32);
                for si in 0..d.scattering_mu_sun as usize {
                    let ms = scattering_mu_from_unit(si as f32 / (d.scattering_mu_sun - 1) as f32);
                    for vi in 0..d.scattering_mu_view as usize {
                        let mv =
                            scattering_mu_from_unit(vi as f32 / (d.scattering_mu_view - 1) as f32);
                        let index = scattering_index(d, hi, ni, si, vi);
                        let sample =
                            integrate_scattering_order(&config, d, &previous, h, mv, ms, nu);
                        next[index] = sample.transport;
                        for w in 0..NUM_WAVELENGTHS {
                            accumulated[index][w] += sample.volume[w];
                        }
                    }
                }
            }
        }
        let delta = next
            .iter()
            .flat_map(|sample| sample.iter())
            .map(|value| value.abs())
            .sum::<f32>()
            / (count * NUM_WAVELENGTHS) as f32;
        deltas.push(delta);
        previous = next;
    }
    let mut single_rgba = Vec::with_capacity(count * 4);
    let mut total_rgba = Vec::with_capacity(count * 4);
    for (s, t) in single.iter().zip(accumulated.iter()) {
        append_rgba(
            &mut single_rgba,
            rgba_from_spectral(s, s.iter().sum::<f32>() / NUM_WAVELENGTHS as f32),
        );
        append_rgba(
            &mut total_rgba,
            rgba_from_spectral(t, t.iter().sum::<f32>() / NUM_WAVELENGTHS as f32),
        );
    }
    let aerial_count =
        d.aerial_distance as usize * d.aerial_mu_view as usize * d.aerial_height as usize;
    let mut aerial = Vec::with_capacity(aerial_count * 4);
    for hi in 0..d.aerial_height {
        let h = config.atmosphere_height_m() * hi as f32 / (d.aerial_height - 1) as f32;
        for vi in 0..d.aerial_mu_view {
            let mu = -1.0 + 2.0 * vi as f32 / (d.aerial_mu_view - 1) as f32;
            for di in 0..d.aerial_distance {
                let distance =
                    config.max_aerial_distance_m * di as f32 / (d.aerial_distance - 1) as f32;
                let t = transmittance_segment(
                    &config,
                    h,
                    mu,
                    distance_to_boundary(&config, h, mu).min(distance),
                );
                let mean = t.iter().sum::<f32>() / NUM_WAVELENGTHS as f32;
                append_rgba(
                    &mut aerial,
                    [f16::ZERO, f16::ZERO, f16::ZERO, f16::from_f32(mean)],
                );
            }
        }
    }
    Ok(AtmosphereLuts {
        metadata: AtmosphereLutMetadata {
            config: config.clone(),
            dimensions: d,
            wavelengths_nm: WAVELENGTHS_NM,
            scattering_orders: config.scattering_orders,
            scattering_lut_semantics: ACCUMULATED_SCATTERING_LUT_SEMANTICS,
            storage_format: "rgba16float",
            precomputed: false,
            precomputed_turbidity_bracket: None,
        },
        transmittance: LutData::new([d.transmittance_mu, d.transmittance_height, 1], trans)?,
        single_scattering: LutData::new(
            [
                d.scattering_mu_view,
                d.scattering_mu_sun,
                d.scattering_height * d.scattering_nu,
            ],
            single_rgba,
        )?,
        multiple_scattering: LutData::new(
            [
                d.scattering_mu_view,
                d.scattering_mu_sun,
                d.scattering_height * d.scattering_nu,
            ],
            total_rgba,
        )?,
        aerial_perspective: LutData::new(
            [d.aerial_distance, d.aerial_mu_view, d.aerial_height],
            aerial,
        )?,
        order_deltas: deltas,
        _sealed_deterministic_sha256: [0; 32],
        _host_visible_allocation: allocation,
    }
    .seal())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> AtmosphereConfig {
        AtmosphereConfig {
            dimensions: LutDimensions {
                transmittance_mu: 8,
                transmittance_height: 4,
                scattering_mu_view: 4,
                scattering_mu_sun: 4,
                scattering_height: 4,
                scattering_nu: 16,
                aerial_distance: 4,
                aerial_mu_view: 4,
                aerial_height: 4,
            },
            ..AtmosphereConfig::default()
        }
    }

    #[test]
    fn nonlinear_scattering_coordinates_roundtrip_and_are_monotonic() {
        let mut pm = -1.0;
        let mut pn = -1.0;
        for i in 0..=1000 {
            let u = i as f32 / 1000.0;
            let m = scattering_mu_from_unit(u);
            let n = scattering_nu_from_unit(u);
            let h = scattering_height_from_unit(u, 100_000.0);
            assert!(m >= pm && n >= pn);
            assert!((scattering_mu_to_unit(m) - u).abs() < 2.0e-6);
            assert!((scattering_nu_to_unit(n) - u).abs() < 1.0e-5);
            assert!((scattering_height_to_unit(h, 100_000.0) - u).abs() < 2.0e-6);
            pm = m;
            pn = n;
        }
        assert_eq!(scattering_mu_from_unit(0.5), 0.0);
        assert!(scattering_mu_from_unit(0.55) < 0.02);
        assert!(scattering_nu_from_unit(0.9) > 0.97);

        let height_nodes: [f32; 8] =
            std::array::from_fn(|index| scattering_height_from_unit(index as f32 / 7.0, 100_000.0));
        assert_eq!(height_nodes[0], 0.0);
        assert_eq!(height_nodes[7], 100_000.0);
        assert!(height_nodes.windows(2).all(|nodes| nodes[1] > nodes[0]));
        assert!((height_nodes[1] - 2_040.816_3).abs() < 0.01);
        assert!((height_nodes[2] - 8_163.265).abs() < 0.01);

        let max_density_interpolation_error = |scale_height: f32, density_aware: bool| {
            let mut maximum = 0.0_f32;
            for sample in 0..=10_000 {
                let height = 100_000.0 * sample as f32 / 10_000.0;
                let unit = if density_aware {
                    scattering_height_to_unit(height, 100_000.0)
                } else {
                    height / 100_000.0
                };
                let coordinate = unit * 7.0;
                let lower = (coordinate.floor() as usize).min(6);
                let fraction = coordinate - lower as f32;
                let node_height = |index: usize| {
                    if density_aware {
                        scattering_height_from_unit(index as f32 / 7.0, 100_000.0)
                    } else {
                        100_000.0 * index as f32 / 7.0
                    }
                };
                let interpolated = (1.0 - fraction) * (-node_height(lower) / scale_height).exp()
                    + fraction * (-node_height(lower + 1) / scale_height).exp();
                maximum = maximum.max((interpolated - (-height / scale_height).exp()).abs());
            }
            maximum
        };
        for scale_height in [1_200.0_f32, 8_000.0] {
            let linear = max_density_interpolation_error(scale_height, false);
            let density_aware = max_density_interpolation_error(scale_height, true);
            assert!(
                density_aware < linear * 0.3,
                "scale={scale_height}, linear={linear}, density_aware={density_aware}"
            );
        }

        let representative_density_error = |density_aware: bool| {
            let mut total = 0.0_f32;
            for scale_height in [1_200.0_f32, 8_000.0] {
                for height in [1_000.0_f32, 10_000.0] {
                    let unit = if density_aware {
                        scattering_height_to_unit(height, 100_000.0)
                    } else {
                        height / 100_000.0
                    };
                    let coordinate = unit * 7.0;
                    let lower = (coordinate.floor() as usize).min(6);
                    let fraction = coordinate - lower as f32;
                    let node_height = |index: usize| {
                        if density_aware {
                            scattering_height_from_unit(index as f32 / 7.0, 100_000.0)
                        } else {
                            100_000.0 * index as f32 / 7.0
                        }
                    };
                    let interpolated = (1.0 - fraction)
                        * (-node_height(lower) / scale_height).exp()
                        + fraction * (-node_height(lower + 1) / scale_height).exp();
                    total += (interpolated - (-height / scale_height).exp()).abs();
                }
            }
            total
        };
        let linear_error = representative_density_error(false);
        let density_aware_error = representative_density_error(true);
        assert!(density_aware_error < linear_error);
        assert!(
            density_aware_error / linear_error < 0.1,
            "linear={linear_error}, density_aware={density_aware_error}"
        );
    }

    #[test]
    fn first_scattering_nu_cell_stays_within_one_percent_of_direct_integration() {
        let c = AtmosphereConfig::default();
        let mu_view = 0.021_696_674_f32;
        let mu_sun = 5.0_f32.to_radians().sin();
        let target_nu = -0.934_005_45_f32;
        let node_count = c.dimensions.scattering_nu;
        let coordinate = scattering_nu_to_unit(target_nu) * (node_count - 1) as f32;
        let lower = coordinate.floor() as u32;
        let fraction = coordinate - lower as f32;
        assert_eq!(
            lower, 0,
            "the residual must exercise the wide first nu cell"
        );

        let integrate_at =
            |nu| integrate_single_scattering(&c, 0.0, mu_view, mu_sun, nu, f32::MAX, 64).0;
        let target = spectral_to_linear_rgb(&integrate_at(target_nu));
        let lower_nu = scattering_nu_from_unit(lower as f32 / (node_count - 1) as f32);
        let upper_nu = scattering_nu_from_unit((lower + 1) as f32 / (node_count - 1) as f32);
        let lower_rgb = spectral_to_linear_rgb(&integrate_at(lower_nu));
        let upper_rgb = spectral_to_linear_rgb(&integrate_at(upper_nu));
        let interpolated = std::array::from_fn::<_, 3, _>(|channel| {
            lower_rgb[channel] * (1.0 - fraction) + upper_rgb[channel] * fraction
        });
        let maximum_relative_error = interpolated
            .iter()
            .zip(target)
            .map(|(actual, expected)| (actual - expected).abs() / expected.abs().max(1.0e-8))
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_relative_error < 0.01,
            "first-cell nu interpolation relative RGB error={maximum_relative_error}"
        );
    }

    #[test]
    fn budget_and_transmittance_contracts() {
        let mut c = AtmosphereConfig::default();
        c.dimensions.aerial_distance = 257;
        assert!(c.validate().is_err());
        let c = small_config();
        let near = transmittance_segment(&c, 0.0, 1.0, 1000.0);
        let far = transmittance_segment(&c, 0.0, 1.0, 40000.0);
        assert!(near.iter().zip(far).all(|(a, b)| b <= *a));
    }

    #[test]
    fn stable_sphere_intersections_classify_near_ground_tangents() {
        let c = AtmosphereConfig::default();
        for height_m in [0.5_f32, 1.0, 10.0] {
            let radius = c.bottom_radius_m + height_m;
            let tangent_mu = (((radius - c.bottom_radius_m) * (radius + c.bottom_radius_m))
                / (radius * radius))
                .sqrt();
            assert!(distance_to_ground(&c, height_m, -tangent_mu * 1.001).is_some());
            assert!(distance_to_ground(&c, height_m, -tangent_mu * 0.999).is_none());
        }
    }

    #[test]
    fn sixty_four_cell_optical_depth_resolves_the_mie_layer() {
        let c = AtmosphereConfig::default();
        let mut max_error_24 = 0.0_f32;
        let mut max_error_64 = 0.0_f32;
        for height_m in [0.0_f32, 1.0, 100.0, 1_200.0, 10_000.0, 50_000.0] {
            for index in 0..=40 {
                let mu = -1.0 + 2.0 * index as f32 / 40.0;
                let distance = distance_to_boundary(&c, height_m, mu);
                let reference = transmittance_from_columns(
                    &c,
                    optical_columns(&c, height_m, mu, distance, 512),
                );
                let coarse =
                    transmittance_from_columns(&c, optical_columns(&c, height_m, mu, distance, 24));
                let production =
                    transmittance_from_columns(&c, optical_columns(&c, height_m, mu, distance, 64));
                for wavelength in 0..NUM_WAVELENGTHS {
                    max_error_24 =
                        max_error_24.max((coarse[wavelength] - reference[wavelength]).abs());
                    max_error_64 =
                        max_error_64.max((production[wavelength] - reference[wavelength]).abs());
                }
            }
        }
        assert!(max_error_64 <= 0.005, "max_error_64={max_error_64}");
        assert!(
            max_error_64 < max_error_24 * 0.25,
            "max_error_24={max_error_24}, max_error_64={max_error_64}"
        );
    }

    #[test]
    fn precomputed_loader_rejects_nearby_physical_values_without_relabeling() {
        let mut config = AtmosphereConfig::default();
        config.ozone_du = 300.00006_f32;
        assert_ne!(config.ozone_du.to_bits(), 300.0_f32.to_bits());
        let error = load_precomputed_atmosphere_luts(config).unwrap_err();
        assert!(matches!(
            error,
            AtmosphereError::UnsupportedPrecomputedConfig(reason)
                if reason.contains("ozone_du")
        ));
    }

    #[test]
    fn precomputed_loader_rejects_noncanonical_dimensions_before_loading() {
        let mut config = AtmosphereConfig::default();
        config.dimensions.transmittance_mu -= 1;
        let error = load_precomputed_atmosphere_luts(config).unwrap_err();
        assert!(matches!(
            error,
            AtmosphereError::UnsupportedPrecomputedConfig(reason)
                if reason.contains("dimensions")
        ));
    }

    #[test]
    fn shared_atmosphere_wgsl_parses_and_keeps_accumulated_contract() {
        let determinism = include_str!("../../shaders/includes/determinism.wgsl");
        let evaluation_core = include_str!("../../shaders/atmosphere/evaluation_core.wgsl");
        let source = include_str!("../../shaders/atmosphere/scattering.wgsl");
        let assembled = format!(
            r#"
struct TestSkyParams {{ sun_size: f32, }}
@group(0) @binding(0) var<uniform> sky_params: TestSkyParams;
fn sky_sun_size(params: TestSkyParams) -> f32 {{ return params.sun_size; }}
struct TestCamera {{ inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>, }}
@group(0) @binding(1) var<uniform> camera: TestCamera;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
{determinism}
{evaluation_core}
{source}
"#
        );
        naga::front::wgsl::parse_str(&assembled).expect("AETHER assembled WGSL must parse");
        for symbol in [
            "struct AtmosphereScatteringUniforms",
            "fn sample_transmittance",
            "fn sample_inscatter",
            "fn sky_radiance",
            "fn phase_rayleigh",
            "fn phase_mie",
        ] {
            assert!(source.contains(symbol), "missing {symbol}");
        }
        for symbol in [
            "fn aether_eval_mu_to_unit",
            "fn aether_eval_nu_to_unit",
            "fn aether_eval_scattering_height_to_unit",
            "fn aether_eval_sample_accumulated_scattering",
            "fn aether_eval_segment_transmittance",
        ] {
            assert!(evaluation_core.contains(symbol), "missing {symbol}");
        }
        assert!(evaluation_core.contains(
            "aether_eval_scattering_height_to_unit(height_unit) * f32(height_count - 1)"
        ));
        assert!(evaluation_core.contains("sqrt(clamp(height_unit, 0.0, 1.0))"));
        assert!(source.contains("sampled exactly once"));
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    fn custom_bake_is_deterministic_and_orders_converge() {
        // Use one extra diagnostic order to bound the unresolved geometric
        // tail below 1%. Shipped N=4 anchors retain their separate strict
        // monotonic lock in
        // `precomputed::tests::every_anchor_decodes_to_finite_complete_payloads`.
        let mut c = small_config();
        c.scattering_orders = 5;
        let a = bake_atmosphere_luts(c.clone()).unwrap();
        assert!(
            a.order_deltas.windows(2).all(|w| w[1] < w[0]),
            "{:?}",
            a.order_deltas
        );
        let max_contraction = a
            .order_deltas
            .windows(2)
            .map(|window| window[1] / window[0])
            .fold(0.0_f32, f32::max);
        assert!(max_contraction < 0.5, "{:?}", a.order_deltas);
        let unresolved_tail_bound =
            a.order_deltas.last().copied().unwrap() * max_contraction / (1.0 - max_contraction);
        assert!(
            unresolved_tail_bound < a.order_deltas[0] * 0.01,
            "deltas={:?}, unresolved_tail_bound={unresolved_tail_bound}",
            a.order_deltas
        );
        assert_ne!(
            a.single_scattering.as_le_bytes(),
            a.multiple_scattering.as_le_bytes()
        );
        assert_eq!(
            a.deterministic_bytes(),
            bake_atmosphere_luts(c).unwrap().deterministic_bytes()
        );
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    fn higher_order_clustered_exact_cells_converge() {
        fn isotropic_transport(
            c: &AtmosphereConfig,
            h: f32,
            mu: f32,
            steps: usize,
            clustered: bool,
        ) -> [f32; NUM_WAVELENGTHS] {
            let length = distance_to_boundary(c, h, mu);
            let ground_bound = distance_to_ground(c, h, mu).is_some();
            let mie = c.mie();
            let mut columns = [0.0; 3];
            let mut result = [0.0; NUM_WAVELENGTHS];
            for index in 0..steps {
                let edge = |cell: usize| {
                    if clustered {
                        assert_eq!(steps, SCATTERING_ORDER_STEPS);
                        scattering_order_cell_edge(length, cell, ground_bound)
                    } else {
                        length * cell as f32 / steps as f32
                    }
                };
                let start = edge(index);
                let end = edge(index + 1);
                let ds = end - start;
                let rho = density_at(c, altitude_along(c, h, mu, 0.5 * (start + end)));
                let view_start = transmittance_from_columns(c, columns);
                for wavelength in 0..NUM_WAVELENGTHS {
                    let wavelength_nm = WAVELENGTHS_NM[wavelength];
                    let isotropic_source = rayleigh_scattering_coefficient(wavelength_nm) * rho[0]
                        + mie.scattering(wavelength_nm) * rho[1];
                    let cell_length =
                        attenuated_cell_length(extinction_at_density(c, rho, wavelength_nm), ds);
                    result[wavelength] += view_start[wavelength] * isotropic_source * cell_length;
                }
                for component in 0..3 {
                    columns[component] += rho[component] * ds;
                }
            }
            result
        }

        let extinction = 0.003_f32;
        let cells = [0.25_f32, 7.0, 1_200.0];
        let mut exact = 0.0;
        let mut distance = 0.0;
        for ds in cells {
            exact += (-extinction * distance).exp() * attenuated_cell_length(extinction, ds);
            distance += ds;
        }
        let closed_form = -(-extinction * distance).exp_m1() / extinction;
        assert!((exact - closed_form).abs() <= closed_form * 2.0e-6);

        let c = AtmosphereConfig::default();
        let cases = [
            (0.0_f32, 1.0_f32),
            (0.0, 0.0),
            (50_000.0, -1.0),
            (c.atmosphere_height_m(), -1.0),
        ];
        let mut max_relative_error = 0.0_f32;
        for (height, mu) in cases {
            let production = isotropic_transport(&c, height, mu, SCATTERING_ORDER_STEPS, true);
            let reference = isotropic_transport(&c, height, mu, 8_192, false);
            for wavelength in 0..NUM_WAVELENGTHS {
                let relative_error = (production[wavelength] - reference[wavelength]).abs()
                    / reference[wavelength].abs().max(1.0e-8);
                max_relative_error = max_relative_error.max(relative_error);
            }
        }
        assert!(
            max_relative_error <= 0.03,
            "max_relative_error={max_relative_error}"
        );
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    fn discrete_quadrature_and_ground_are_physical() {
        let horizon_geometry =
            ray_sample_geometry(&AtmosphereConfig::default(), 1.0, 0.0, 0.0, 1.0, 50_000.0);
        assert!(horizon_geometry.altitude_m > 1.0);
        assert!(horizon_geometry.mu_sun > 0.0, "{horizon_geometry:?}");

        let omega = angular_quadrature().iter().map(|(_, w)| w).sum::<f32>();
        assert!((omega - 4.0 * std::f32::consts::PI).abs() < 1.0e-5);
        assert_eq!(angular_quadrature().len(), 16 * 32);
        assert!(
            angular_quadrature()
                .iter()
                .map(|(local, _)| local[1].abs())
                .fold(f32::INFINITY, f32::min)
                < 0.1
        );
        let phase = angular_quadrature()
            .iter()
            .map(|(v, w)| rayleigh_phase(v[1]) * w)
            .sum::<f32>();
        assert!((phase - 1.0).abs() < 1.0e-5);
        let ground_cosine_weight = angular_quadrature()
            .iter()
            .filter(|(local, _)| local[1] > 0.0)
            .map(|(local, omega)| local[1] * omega)
            .sum::<f32>();
        let ground_normalization = std::f32::consts::PI / ground_cosine_weight;
        let normalized_ground_weight = angular_quadrature()
            .iter()
            .filter(|(local, _)| local[1] > 0.0)
            .map(|(local, omega)| local[1] * omega * ground_normalization)
            .sum::<f32>();
        assert!((normalized_ground_weight - std::f32::consts::PI).abs() < 1.0e-5);
        let unit_albedo_constant_field_gain = normalized_ground_weight / std::f32::consts::PI;
        assert!((unit_albedo_constant_field_gain - 1.0).abs() < 1.0e-5);
        let outgoing_directions = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
            ],
            angular_quadrature()[0].0,
        ];
        for outgoing in outgoing_directions {
            let geometry = RaySampleGeometry {
                altitude_m: 0.0,
                mu_sun: 1.0,
                outgoing,
                sun: [0.0, 1.0, 0.0],
                up: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
            };
            let normalization = phase_quadrature_normalization(geometry, 0.8);
            assert!(
                (0.99..=1.01).contains(&normalization[0])
                    && (0.99..=1.01).contains(&normalization[1]),
                "raw phase integral drifted: {normalization:?}"
            );
            let mut normalized = [0.0_f32; 2];
            for &(local, omega) in angular_quadrature() {
                let cosine = dot(quadrature_direction(geometry, local), outgoing);
                normalized[0] += rayleigh_phase(cosine) * omega / normalization[0];
                normalized[1] += mie_phase_cornette_shanks(cosine, 0.8) * omega / normalization[1];
            }
            assert!((normalized[0] - 1.0).abs() < 1.0e-5, "{normalized:?}");
            assert!((normalized[1] - 1.0).abs() < 1.0e-5, "{normalized:?}");
        }

        // A wavelength-dependent horizon field is the adversarial case the
        // old octahedral rule missed. Compare the actual normalized gather to
        // a much denser midpoint sphere oracle; the narrow "blue" field must
        // not disappear between the vertical ordinates.
        let gather_geometry = RaySampleGeometry {
            altitude_m: 1.0,
            mu_sun: 10.0_f32.to_radians().sin(),
            outgoing: [0.991_399_4, 0.021_515_133, 0.129_090_8],
            sun: [0.925_416_5, 0.173_648_18, 0.336_824_06],
            up: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        };
        let normalization = phase_quadrature_normalization(gather_geometry, 0.8);
        for width in [0.22_f32, 0.55_f32] {
            let mut production = 0.0_f32;
            for &(local, omega) in angular_quadrature() {
                let incoming = quadrature_direction(gather_geometry, local);
                let cosine = dot(incoming, gather_geometry.outgoing);
                let incident = (-(local[1] / width).powi(2)).exp();
                production += incident
                    * (rayleigh_phase(cosine) / normalization[0]
                        + mie_phase_cornette_shanks(cosine, 0.8) / normalization[1])
                    * omega;
            }
            let mut oracle = 0.0_f32;
            const ORACLE_MU: usize = 128;
            const ORACLE_AZIMUTH: usize = 256;
            let oracle_weight = 4.0 * std::f32::consts::PI / (ORACLE_MU * ORACLE_AZIMUTH) as f32;
            for mu_index in 0..ORACLE_MU {
                let mu = -1.0 + 2.0 * (mu_index as f32 + 0.5) / ORACLE_MU as f32;
                let radial = (1.0 - mu * mu).sqrt();
                let incident = (-(mu / width).powi(2)).exp();
                for azimuth_index in 0..ORACLE_AZIMUTH {
                    let phi = std::f32::consts::TAU * (azimuth_index as f32 + 0.5)
                        / ORACLE_AZIMUTH as f32;
                    let local = [radial * phi.cos(), mu, radial * phi.sin()];
                    let incoming = quadrature_direction(gather_geometry, local);
                    let cosine = dot(incoming, gather_geometry.outgoing);
                    oracle += incident
                        * (rayleigh_phase(cosine) + mie_phase_cornette_shanks(cosine, 0.8))
                        * oracle_weight;
                }
            }
            let relative_error = (production - oracle).abs() / oracle;
            assert!(
                relative_error < 0.01,
                "width={width}, production={production}, oracle={oracle}, relative_error={relative_error}"
            );
        }
        let mut dark = small_config();
        let zero_previous = vec![
            [0.0; NUM_WAVELENGTHS];
            dark.dimensions.scattering_height as usize
                * dark.dimensions.scattering_nu as usize
                * dark.dimensions.scattering_mu_sun as usize
                * dark.dimensions.scattering_mu_view as usize
        ];
        let sun = [
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ];
        let mut direct_config = dark.clone();
        direct_config.ground_albedo = 0.8;
        let direct =
            ground_boundary_source(&direct_config, direct_config.dimensions, None, sun, true);
        assert!(direct.iter().all(|value| *value > 0.0), "{direct:?}");
        let ground_seed = ground_boundary_along_ray(
            &direct_config,
            direct_config.dimensions,
            None,
            0.0,
            -1.0,
            sun[1],
            -sun[1],
            true,
        );
        assert!(
            ground_seed.iter().all(|value| *value > 0.0),
            "zero-length downward rays must retain the direct boundary: {ground_seed:?}"
        );
        assert_eq!(
            ground_boundary_source(
                &direct_config,
                direct_config.dimensions,
                Some(&zero_previous),
                sun,
                false,
            ),
            [0.0; NUM_WAVELENGTHS]
        );
        direct_config.ground_albedo = 0.0;
        assert_eq!(
            ground_boundary_source(&direct_config, direct_config.dimensions, None, sun, true,),
            [0.0; NUM_WAVELENGTHS]
        );

        // F1 is atmospheric single scattering plus the direct Lambertian
        // boundary. Consequently order 2, not order 3, already scatters that
        // ground energy into above-horizon sky.
        dark.scattering_orders = 2;
        dark.ground_albedo = 0.0;
        let mut bright = dark.clone();
        bright.ground_albedo = 0.8;
        let d = bake_atmosphere_luts(dark).unwrap();
        let b = bake_atmosphere_luts(bright).unwrap();
        let higher = |x: &AtmosphereLuts| {
            x.multiple_scattering
                .rgba_f32()
                .into_iter()
                .zip(x.single_scattering.rgba_f32())
                .map(|(a, s)| (a - s).max(0.0))
                .sum::<f32>()
        };
        assert!(higher(&b) > higher(&d));
        let view = [(1.0_f32 - 0.2_f32.powi(2)).sqrt(), 0.2, 0.0];
        let dark_sky = d.sky_radiance(0.0, view, sun).unwrap();
        let bright_sky = b.sky_radiance(0.0, view, sun).unwrap();
        assert!(
            bright_sky.iter().sum::<f32>() > dark_sky.iter().sum::<f32>(),
            "dark={dark_sky:?}, bright={bright_sky:?}"
        );
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    fn aerial_payload_is_transmittance_only_and_dynamic() {
        let l = bake_atmosphere_luts(small_config()).unwrap();
        assert!(l
            .aerial_perspective
            .texels
            .chunks_exact(4)
            .all(|v| v[0] == f16::ZERO && v[1] == f16::ZERO && v[2] == f16::ZERO));
        let view = [
            0.8,
            0.3,
            (1.0_f32 - 0.8_f32.powi(2) - 0.3_f32.powi(2)).sqrt(),
        ];
        let a = l
            .apply_aerial_perspective([0.0; 3], 500.0, 40000.0, view, [0.8660254, 0.5, 0.0])
            .unwrap();
        let b = l
            .apply_aerial_perspective([0.0; 3], 500.0, 40000.0, view, [-0.8660254, 0.5, 0.0])
            .unwrap();
        assert_ne!(a, b);
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    #[ignore = "maintainer-only full-bank bake; routine CI locks anchor SHA256 and small-bake convergence"]
    fn shipped_anchor_payloads_match_fresh_bakes() {
        for turbidity in TURBIDITY_BANK {
            let mut c = AtmosphereConfig::default();
            c.turbidity = turbidity;
            let baked = bake_atmosphere_luts(c.clone()).unwrap();
            let shipped = load_precomputed_atmosphere_luts(c).unwrap();
            let close = |a: &LutData, b: &LutData| {
                assert_eq!(a.dimensions, b.dimensions);
                assert!(a
                    .texels
                    .iter()
                    .zip(&b.texels)
                    .all(|(x, y)| x.to_bits().abs_diff(y.to_bits()) <= 4));
            };
            close(&shipped.transmittance, &baked.transmittance);
            close(&shipped.single_scattering, &baked.single_scattering);
            close(&shipped.multiple_scattering, &baked.multiple_scattering);
            close(&shipped.aerial_perspective, &baked.aerial_perspective);
            for (a, b) in shipped.order_deltas.iter().zip(&baked.order_deltas) {
                assert!((a - b).abs() <= 1.0e-8_f32.max(b.abs() * 1.0e-5));
            }
        }
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    #[ignore]
    fn regenerate_exact_precomputed_bank_for_maintainers() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/core/atmosphere/precomputed");
        std::fs::create_dir_all(&dir).unwrap();
        for turbidity in TURBIDITY_BANK {
            let mut c = AtmosphereConfig::default();
            c.turbidity = turbidity;
            let l = bake_atmosphere_luts(c).unwrap();
            let mut bytes = Vec::with_capacity(precomputed::ANCHOR_BYTES);
            bytes.extend_from_slice(&l.transmittance.as_le_bytes());
            bytes.extend_from_slice(&l.single_scattering.as_le_bytes());
            bytes.extend_from_slice(&l.multiple_scattering.as_le_bytes());
            bytes.extend_from_slice(&l.aerial_perspective.as_le_bytes());
            for d in &l.order_deltas {
                bytes.extend_from_slice(&d.to_le_bytes());
            }
            assert_eq!(bytes.len(), precomputed::ANCHOR_BYTES);
            let path = dir.join(format!("turbidity-{turbidity:.0}.bin"));
            std::fs::write(&path, bytes).unwrap();
            println!("wrote {} {:?}", path.display(), l.order_deltas);
        }
    }
}
