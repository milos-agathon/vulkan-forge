//! Exact, offline-baked atmosphere anchors shipped with the runtime.

use half::f16;

pub(super) const TURBIDITY_BANK: [f32; 5] = [1.0, 2.0, 4.0, 8.0, 10.0];
pub(super) const TRANSMITTANCE_DIMENSIONS: [u32; 3] = [32, 8, 1];
pub(super) const SCATTERING_DIMENSIONS: [u32; 3] = [17, 17, 8 * 16];
pub(super) const AERIAL_DIMENSIONS: [u32; 3] = [8, 8, 8];
pub(super) const SCATTERING_HEIGHT: u32 = 8;
pub(super) const SCATTERING_NU: u32 = 16;
pub(super) const SCATTERING_ORDERS: usize = 4;

const RGBA_CHANNELS: usize = 4;
const F16_BYTES: usize = 2;
const TRANSMITTANCE_BYTES: usize = 32 * 8 * RGBA_CHANNELS * F16_BYTES;
const SCATTERING_BYTES: usize = 17 * 17 * 8 * 16 * RGBA_CHANNELS * F16_BYTES;
const AERIAL_BYTES: usize = 8 * 8 * 8 * RGBA_CHANNELS * F16_BYTES;
const ORDER_DELTA_BYTES: usize = SCATTERING_ORDERS * std::mem::size_of::<f32>();
const TRANSMITTANCE_OFFSET: usize = 0;
const SINGLE_SCATTERING_OFFSET: usize = TRANSMITTANCE_OFFSET + TRANSMITTANCE_BYTES;
const ACCUMULATED_SCATTERING_OFFSET: usize = SINGLE_SCATTERING_OFFSET + SCATTERING_BYTES;
const AERIAL_OFFSET: usize = ACCUMULATED_SCATTERING_OFFSET + SCATTERING_BYTES;
const ORDER_DELTAS_OFFSET: usize = AERIAL_OFFSET + AERIAL_BYTES;
pub(super) const ANCHOR_BYTES: usize = ORDER_DELTAS_OFFSET + ORDER_DELTA_BYTES;

const ANCHOR_1: &[u8; ANCHOR_BYTES] = include_bytes!("precomputed/turbidity-1.bin");
const ANCHOR_2: &[u8; ANCHOR_BYTES] = include_bytes!("precomputed/turbidity-2.bin");
const ANCHOR_4: &[u8; ANCHOR_BYTES] = include_bytes!("precomputed/turbidity-4.bin");
const ANCHOR_8: &[u8; ANCHOR_BYTES] = include_bytes!("precomputed/turbidity-8.bin");
const ANCHOR_10: &[u8; ANCHOR_BYTES] = include_bytes!("precomputed/turbidity-10.bin");
const ANCHORS: [&[u8; ANCHOR_BYTES]; 5] = [ANCHOR_1, ANCHOR_2, ANCHOR_4, ANCHOR_8, ANCHOR_10];

#[cfg(test)]
const ANCHOR_SHA256: [&str; 5] = [
    "9ead28087343283942d0bf834aecfb7b3a7b0ea513b830731c2cf9bc77a15f0b",
    "c6a77bd25241d6123078cace17d9a2181b520c44ac0e871274d755e092e565bc",
    "350a1d13863ac0f4a38a3be585e663a8e8c701c14cb5484760cb0d5ccbe772cd",
    "56594423699db4a644650e21f231824f19cb52c0abd718c88b7e4f21f00759cf",
    "633b77f0a6d8c31a4640e666ba45c7068fa31b7f623b1f711e5117583c1f51f5",
];

pub(super) struct InterpolatedPayload {
    pub(super) transmittance: Vec<f16>,
    pub(super) single_scattering: Vec<f16>,
    pub(super) accumulated_scattering: Vec<f16>,
    pub(super) aerial_perspective: Vec<f16>,
    pub(super) order_deltas: Vec<f32>,
}

fn selected_anchor(lower: usize, upper: usize, factor: f32) -> Option<usize> {
    if lower == upper || factor <= 0.0 {
        Some(lower)
    } else if factor >= 1.0 {
        Some(upper)
    } else {
        None
    }
}

fn interpolate_f16(
    offset: usize,
    byte_len: usize,
    lower: usize,
    upper: usize,
    factor: f32,
) -> Vec<f16> {
    if let Some(anchor) = selected_anchor(lower, upper, factor) {
        return ANCHORS[anchor][offset..offset + byte_len]
            .chunks_exact(F16_BYTES)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect();
    }
    ANCHORS[lower][offset..offset + byte_len]
        .chunks_exact(F16_BYTES)
        .zip(ANCHORS[upper][offset..offset + byte_len].chunks_exact(F16_BYTES))
        .map(|(a, b)| {
            let a = f16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
            let b = f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            f16::from_f32(a + (b - a) * factor)
        })
        .collect()
}

fn interpolate_order_deltas(lower: usize, upper: usize, factor: f32) -> Vec<f32> {
    let decode = |anchor: usize, order: usize| {
        let offset = ORDER_DELTAS_OFFSET + order * 4;
        f32::from_le_bytes(ANCHORS[anchor][offset..offset + 4].try_into().unwrap())
    };
    (0..SCATTERING_ORDERS)
        .map(|order| match selected_anchor(lower, upper, factor) {
            Some(anchor) => decode(anchor, order),
            None => {
                let a = decode(lower, order);
                a + (decode(upper, order) - a) * factor
            }
        })
        .collect()
}

pub(super) fn interpolate(lower: usize, upper: usize, factor: f32) -> InterpolatedPayload {
    InterpolatedPayload {
        transmittance: interpolate_f16(
            TRANSMITTANCE_OFFSET,
            TRANSMITTANCE_BYTES,
            lower,
            upper,
            factor,
        ),
        single_scattering: interpolate_f16(
            SINGLE_SCATTERING_OFFSET,
            SCATTERING_BYTES,
            lower,
            upper,
            factor,
        ),
        accumulated_scattering: interpolate_f16(
            ACCUMULATED_SCATTERING_OFFSET,
            SCATTERING_BYTES,
            lower,
            upper,
            factor,
        ),
        aerial_perspective: interpolate_f16(AERIAL_OFFSET, AERIAL_BYTES, lower, upper, factor),
        order_deltas: interpolate_order_deltas(lower, upper, factor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn exact_anchor_assets_match_locked_sha256() {
        for (index, expected) in ANCHOR_SHA256.iter().enumerate() {
            assert_eq!(ANCHORS[index].len(), ANCHOR_BYTES);
            assert_eq!(format!("{:x}", Sha256::digest(ANCHORS[index])), *expected);
        }
    }

    #[test]
    fn every_anchor_decodes_to_finite_complete_payloads() {
        for index in 0..ANCHORS.len() {
            let payload = interpolate(index, index, 0.0);
            assert_eq!(payload.transmittance.len(), TRANSMITTANCE_BYTES / F16_BYTES);
            assert_eq!(
                payload.single_scattering.len(),
                SCATTERING_BYTES / F16_BYTES
            );
            assert_eq!(
                payload.accumulated_scattering.len(),
                SCATTERING_BYTES / F16_BYTES
            );
            assert_eq!(payload.aerial_perspective.len(), AERIAL_BYTES / F16_BYTES);
            assert_eq!(payload.order_deltas.len(), SCATTERING_ORDERS);
            assert!(payload
                .transmittance
                .iter()
                .chain(&payload.single_scattering)
                .chain(&payload.accumulated_scattering)
                .chain(&payload.aerial_perspective)
                .all(|value| value.is_finite()));
            assert!(payload
                .order_deltas
                .iter()
                .all(|value| value.is_finite() && *value > 0.0));
            assert!(payload
                .order_deltas
                .windows(2)
                .all(|pair| pair[1] < pair[0]));
        }
    }
}
