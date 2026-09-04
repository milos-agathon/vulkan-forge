//! Deterministic BC5 UNORM encoder/decoder.
//!
//! Each 4x4 block is two independently encoded BC4 blocks (R then G). The
//! implementation intentionally operates on RG8 input so normal maps do not
//! carry a redundant reconstructed Z channel through the packed store.

use super::bc4;

pub fn encode_bc5_rg8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let expected = width as usize * height as usize * 2;
    if data.len() != expected {
        return Err(format!(
            "BC5 input mismatch: got {} bytes, expected {expected}",
            data.len()
        ));
    }
    let red = bc4::encode_channel(data, width, height, 2, 0)?;
    let green = bc4::encode_channel(data, width, height, 2, 1)?;
    let blocks = width.div_ceil(4) as usize * height.div_ceil(4) as usize;
    let mut encoded = Vec::with_capacity(blocks * 16);
    for block in 0..blocks {
        encoded.extend_from_slice(&red[block * 8..block * 8 + 8]);
        encoded.extend_from_slice(&green[block * 8..block * 8 + 8]);
    }
    Ok(encoded)
}

pub fn decode_bc5_rg8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks = width.div_ceil(4) as usize * height.div_ceil(4) as usize;
    if data.len() != blocks * 16 {
        return Err(format!(
            "BC5 block payload mismatch: got {} bytes, expected {}",
            data.len(),
            blocks * 16
        ));
    }
    let mut red = Vec::with_capacity(blocks * 8);
    let mut green = Vec::with_capacity(blocks * 8);
    for block in 0..blocks {
        red.extend_from_slice(&data[block * 16..block * 16 + 8]);
        green.extend_from_slice(&data[block * 16 + 8..block * 16 + 16]);
    }
    let red = bc4::decode_channel(&red, width, height)?;
    let green = bc4::decode_channel(&green, width, height)?;
    let mut decoded = Vec::with_capacity(width as usize * height as usize * 2);
    for (&r, &g) in red.iter().zip(&green) {
        decoded.extend_from_slice(&[r, g]);
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_normal_round_trips_exactly() {
        let mut rg = Vec::new();
        for _ in 0..64 {
            rg.extend_from_slice(&[128, 128]);
        }
        let encoded = encode_bc5_rg8(&rg, 8, 8).unwrap();
        assert_eq!(encoded.len(), 64);
        assert_eq!(decode_bc5_rg8(&encoded, 8, 8).unwrap(), rg);
    }

    #[test]
    fn decodes_spec_constructed_bc5_reference_block() {
        // Two independently constructed BC4 reference blocks: solid R=255
        // followed by solid G=0, in the BC5-mandated R-then-G ordering.
        let reference = [255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            decode_bc5_rg8(&reference, 4, 4).unwrap(),
            [255, 0].repeat(16)
        );
    }
}
