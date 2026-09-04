//! Typed handoff from an AETHER LUT bake to active renderer consumers.

use std::sync::Arc;

use super::{
    load_precomputed_atmosphere_luts, AtmosphereConfig, AtmosphereError, AtmosphereLuts,
    LutDimensions, ACCUMULATED_SCATTERING_LUT_SEMANTICS, AETHER_RADIOMETRIC_SCALE_MAX,
    WAVELENGTHS_NM,
};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_host_allocation, ResourceHandle};

/// Exact little-endian upload encoding with scoped host-visible accounting.
pub(crate) struct TrackedLutUploadBytes {
    bytes: Vec<u8>,
    _allocation: ResourceHandle,
}

impl TrackedLutUploadBytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) fn tracked_lut_upload_bytes(
    data: &super::LutData,
    label: &str,
) -> Result<TrackedLutUploadBytes, RenderError> {
    let expected = data.byte_size();
    let allocation = tracked_host_allocation(expected, label)?;
    let bytes = data.as_le_bytes();
    debug_assert_eq!(bytes.len() as u64, expected);
    Ok(TrackedLutUploadBytes {
        bytes,
        _allocation: allocation,
    })
}

/// Immutable, provenance-keyed atmosphere transport consumed by GPU runtimes.
///
/// The handle owns the tracked host-visible LUT allocation through an `Arc`.
/// Cloning it therefore transfers the exact baked payload without copying it or
/// re-resolving a nearby shipped table.
#[derive(Clone, Debug)]
pub struct AtmosphereLutHandle {
    luts: Arc<AtmosphereLuts>,
    deterministic_sha256: [u8; 32],
}

impl AtmosphereLutHandle {
    /// Validate and adopt one exact baked or precomputed LUT payload.
    pub fn from_luts(luts: AtmosphereLuts) -> Result<Self, AtmosphereError> {
        validate_runtime_luts(&luts)?;
        let deterministic_sha256 = luts.deterministic_sha256();
        if deterministic_sha256 != luts.sealed_deterministic_sha256() {
            return Err(AtmosphereError::InvalidConfig(
                "LUT payload or physical metadata changed after bake/load; refusing relabeled provenance"
                    .into(),
            ));
        }
        Ok(Self {
            luts: Arc::new(luts),
            deterministic_sha256,
        })
    }

    /// Resolve the provenance-locked shipped bank into the same typed handoff.
    pub fn load_shipped(config: AtmosphereConfig) -> Result<Self, AtmosphereError> {
        Self::from_luts(load_precomputed_atmosphere_luts(config)?)
    }

    pub fn config(&self) -> &AtmosphereConfig {
        &self.luts.metadata.config
    }

    pub fn luts(&self) -> &AtmosphereLuts {
        &self.luts
    }

    pub fn deterministic_sha256(&self) -> [u8; 32] {
        self.deterministic_sha256
    }

    pub fn deterministic_sha256_hex(&self) -> String {
        self.deterministic_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn expected_dimensions(dimensions: LutDimensions) -> Result<[[u32; 3]; 4], AtmosphereError> {
    let scattering_depth = dimensions
        .scattering_height
        .checked_mul(dimensions.scattering_nu)
        .ok_or(AtmosphereError::DimensionOverflow)?;
    Ok([
        [
            dimensions.transmittance_mu,
            dimensions.transmittance_height,
            1,
        ],
        [
            dimensions.scattering_mu_view,
            dimensions.scattering_mu_sun,
            scattering_depth,
        ],
        [
            dimensions.scattering_mu_view,
            dimensions.scattering_mu_sun,
            scattering_depth,
        ],
        [
            dimensions.aerial_distance,
            dimensions.aerial_mu_view,
            dimensions.aerial_height,
        ],
    ])
}

fn validate_lut_payload(
    label: &str,
    data: &super::LutData,
    expected_dimensions: [u32; 3],
    max_value: f32,
) -> Result<(), AtmosphereError> {
    let expected_components = expected_dimensions
        .into_iter()
        .try_fold(4_u64, |count, axis| count.checked_mul(u64::from(axis)))
        .ok_or(AtmosphereError::DimensionOverflow)?;
    if data.texels.len() as u64 != expected_components {
        return Err(AtmosphereError::InvalidConfig(format!(
            "runtime {label} payload has {} components, expected {expected_components}",
            data.texels.len()
        )));
    }
    if let Some((index, value)) = data
        .texels
        .iter()
        .map(|value| value.to_f32())
        .enumerate()
        .find(|(_, value)| !value.is_finite() || !(0.0..=max_value).contains(value))
    {
        return Err(AtmosphereError::InvalidConfig(format!(
            "runtime {label} payload component {index} must be finite and in [0, {max_value}], got {value}"
        )));
    }
    Ok(())
}

fn validate_runtime_luts(luts: &AtmosphereLuts) -> Result<(), AtmosphereError> {
    luts.metadata.config.validate()?;
    if luts.metadata.config.dimensions != luts.metadata.dimensions {
        return Err(AtmosphereError::InvalidConfig(
            "LUT metadata dimensions do not match its physical configuration".into(),
        ));
    }
    if luts.metadata.scattering_orders != luts.metadata.config.scattering_orders {
        return Err(AtmosphereError::InvalidConfig(
            "LUT scattering order metadata does not match its physical configuration".into(),
        ));
    }
    if luts.metadata.wavelengths_nm != WAVELENGTHS_NM {
        return Err(AtmosphereError::InvalidConfig(
            "runtime LUT wavelength basis does not match AETHER's canonical 11 wavelengths".into(),
        ));
    }
    match (
        luts.metadata.precomputed,
        luts.metadata.precomputed_turbidity_bracket,
    ) {
        (true, Some([lower, upper]))
            if lower.is_finite()
                && upper.is_finite()
                && (1.0..=10.0).contains(&lower)
                && (lower..=10.0).contains(&upper)
                && (lower..=upper).contains(&luts.metadata.config.turbidity) => {}
        (false, None) => {}
        _ => {
            return Err(AtmosphereError::InvalidConfig(
                "runtime LUT precomputed provenance and turbidity bracket are inconsistent".into(),
            ));
        }
    }
    if luts.metadata.storage_format != "rgba16float" {
        return Err(AtmosphereError::InvalidConfig(format!(
            "runtime LUT storage must be rgba16float, got {}",
            luts.metadata.storage_format
        )));
    }
    if luts.metadata.scattering_lut_semantics != ACCUMULATED_SCATTERING_LUT_SEMANTICS {
        return Err(AtmosphereError::InvalidConfig(format!(
            "runtime scattering semantics must be {ACCUMULATED_SCATTERING_LUT_SEMANTICS}, got {}",
            luts.metadata.scattering_lut_semantics
        )));
    }
    let expected = expected_dimensions(luts.metadata.dimensions)?;
    let actual = [
        luts.transmittance.dimensions,
        luts.single_scattering.dimensions,
        luts.multiple_scattering.dimensions,
        luts.aerial_perspective.dimensions,
    ];
    if actual != expected {
        return Err(AtmosphereError::InvalidConfig(format!(
            "runtime LUT payload dimensions {actual:?} do not match metadata {expected:?}"
        )));
    }
    for (label, data, dimensions, max_value) in [
        ("transmittance", &luts.transmittance, expected[0], 1.0),
        (
            "single-scattering",
            &luts.single_scattering,
            expected[1],
            AETHER_RADIOMETRIC_SCALE_MAX,
        ),
        (
            "accumulated-scattering",
            &luts.multiple_scattering,
            expected[2],
            AETHER_RADIOMETRIC_SCALE_MAX,
        ),
        (
            "aerial-perspective",
            &luts.aerial_perspective,
            expected[3],
            1.0,
        ),
    ] {
        validate_lut_payload(label, data, dimensions, max_value)?;
    }
    if luts.aerial_perspective.texels.chunks_exact(4).any(|rgba| {
        rgba[0].to_f32() != 0.0
            || rgba[1].to_f32() != 0.0
            || rgba[2].to_f32() != 0.0
            || !(0.0..=1.0).contains(&rgba[3].to_f32())
    }) {
        return Err(AtmosphereError::InvalidConfig(
            "runtime aerial-perspective payload must store zero RGB and unit-bounded transmittance alpha"
                .into(),
        ));
    }
    if luts.order_deltas.len() != luts.metadata.scattering_orders as usize
        || luts
            .order_deltas
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(AtmosphereError::InvalidConfig(
            "runtime scattering-order deltas must contain one finite nonnegative value per order"
                .into(),
        ));
    }
    if luts.byte_size() != luts.metadata.dimensions.payload_bytes()? {
        return Err(AtmosphereError::InvalidConfig(
            "runtime LUT byte size does not match its tracked metadata budget".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn shipped_handle_retains_exact_config_and_stable_key() {
        let config = AtmosphereConfig::default();
        let first = AtmosphereLutHandle::load_shipped(config.clone()).unwrap();
        let second = AtmosphereLutHandle::load_shipped(config.clone()).unwrap();
        assert_eq!(first.config(), &config);
        assert!(first.luts().metadata.precomputed);
        assert_eq!(first.deterministic_sha256(), second.deterministic_sha256());
        let legacy_digest: [u8; 32] = Sha256::digest(first.luts().deterministic_bytes()).into();
        assert_eq!(first.deterministic_sha256(), legacy_digest);
        assert_eq!(first.deterministic_sha256_hex().len(), 64);
    }

    #[cfg(feature = "atmosphere-bake")]
    #[test]
    fn custom_bake_handle_retains_supported_physical_inputs() {
        let mut config = AtmosphereConfig {
            ozone_du: 321.0,
            mie_g: 0.71,
            ground_albedo: 0.42,
            scattering_orders: 2,
            ..AtmosphereConfig::default()
        };
        config.dimensions = LutDimensions {
            transmittance_mu: 2,
            transmittance_height: 2,
            scattering_mu_view: 2,
            scattering_mu_sun: 2,
            scattering_height: 2,
            scattering_nu: 2,
            aerial_distance: 2,
            aerial_mu_view: 2,
            aerial_height: 2,
        };
        let handle = AtmosphereLutHandle::from_luts(
            crate::core::atmosphere::bake_atmosphere_luts(config.clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(handle.config(), &config);
        assert!(!handle.luts().metadata.precomputed);
    }

    #[test]
    fn upload_encoding_is_scoped_and_host_visible_tracked() {
        let handle = AtmosphereLutHandle::load_shipped(Default::default()).unwrap();
        let data = &handle.luts().transmittance;
        let label = "aether-test-lut-upload-staging";
        let staged = tracked_lut_upload_bytes(data, label).unwrap();
        assert_eq!(staged.as_slice().len() as u64, data.byte_size());
        assert_eq!(
            crate::core::resource_tracker::ledger_snapshot()
                .by_label
                .get(label),
            Some(&data.byte_size())
        );
        drop(staged);
        assert!(!crate::core::resource_tracker::ledger_snapshot()
            .by_label
            .contains_key(label));
    }

    #[test]
    fn handle_rejects_mutated_lut_payload_values_and_shapes() {
        let mut non_finite = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        non_finite.multiple_scattering.texels[0] = half::f16::NAN;
        assert!(AtmosphereLutHandle::from_luts(non_finite).is_err());

        let mut infinite = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        infinite.single_scattering.texels[0] = half::f16::from_f32(f32::INFINITY);
        assert!(AtmosphereLutHandle::from_luts(infinite).is_err());

        let mut negative = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        negative.single_scattering.texels[0] = half::f16::from_f32(-1.0);
        assert!(AtmosphereLutHandle::from_luts(negative).is_err());

        let mut nonphysical_transmittance =
            load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        nonphysical_transmittance.transmittance.texels[0] = half::f16::from_f32(2.0);
        assert!(AtmosphereLutHandle::from_luts(nonphysical_transmittance).is_err());

        let mut reshaped = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        let moved = reshaped.multiple_scattering.texels.pop().unwrap();
        reshaped.transmittance.texels.push(moved);
        assert_eq!(
            reshaped.byte_size(),
            reshaped.metadata.dimensions.payload_bytes().unwrap()
        );
        assert!(AtmosphereLutHandle::from_luts(reshaped).is_err());
    }

    #[test]
    fn handle_rejects_mutated_aerial_semantics_and_order_deltas() {
        let mut aerial = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        aerial.aerial_perspective.texels[0] = half::f16::from_f32(0.25);
        assert!(AtmosphereLutHandle::from_luts(aerial).is_err());

        let mut deltas = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        deltas.order_deltas[0] = f32::INFINITY;
        assert!(AtmosphereLutHandle::from_luts(deltas).is_err());
    }

    #[test]
    fn handle_rejects_valid_range_payload_and_metadata_relabeling() {
        let mut payload = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        let original_payload_key = payload.deterministic_sha256();
        payload.multiple_scattering.texels[0] = half::f16::from_f32(0.5);
        assert_ne!(payload.deterministic_sha256(), original_payload_key);
        assert!(AtmosphereLutHandle::from_luts(payload).is_err());

        let mut config = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        let original_config_key = config.deterministic_sha256();
        config.metadata.config.ozone_du = 301.0;
        assert_ne!(config.deterministic_sha256(), original_config_key);
        assert!(AtmosphereLutHandle::from_luts(config).is_err());

        let mut spectral = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        let original_spectral_key = spectral.deterministic_sha256();
        spectral.metadata.wavelengths_nm[0] += 1.0;
        assert_ne!(spectral.deterministic_sha256(), original_spectral_key);
        assert!(AtmosphereLutHandle::from_luts(spectral).is_err());

        let mut provenance = load_precomputed_atmosphere_luts(AtmosphereConfig::default()).unwrap();
        let original_provenance_key = provenance.deterministic_sha256();
        provenance.metadata.precomputed = false;
        provenance.metadata.precomputed_turbidity_bracket = None;
        assert_ne!(provenance.deterministic_sha256(), original_provenance_key);
        assert!(AtmosphereLutHandle::from_luts(provenance).is_err());
    }
}
