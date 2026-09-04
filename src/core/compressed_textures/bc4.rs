//! Deterministic BC4 UNORM block codec used by the BC5 normal encoder.

const BLOCK_BYTES: usize = 8;

pub(crate) fn encode_channel(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    channel: usize,
) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("BC4 dimensions must be non-zero".to_string());
    }
    let expected = width as usize * height as usize * stride;
    if data.len() != expected || channel >= stride {
        return Err(format!(
            "BC4 input mismatch: got {} bytes, expected {expected} (stride={stride}, channel={channel})",
            data.len()
        ));
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut encoded = Vec::with_capacity((blocks_x * blocks_y) as usize * BLOCK_BYTES);
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let mut values = [0u8; 16];
            for local_y in 0..4 {
                for local_x in 0..4 {
                    let x = (block_x * 4 + local_x).min(width - 1);
                    let y = (block_y * 4 + local_y).min(height - 1);
                    values[(local_y * 4 + local_x) as usize] =
                        data[(y as usize * width as usize + x as usize) * stride + channel];
                }
            }
            encoded.extend_from_slice(&encode_block(&values));
        }
    }
    Ok(encoded)
}

pub(crate) fn decode_channel(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("BC4 dimensions must be non-zero".to_string());
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let expected = (blocks_x * blocks_y) as usize * BLOCK_BYTES;
    if data.len() != expected {
        return Err(format!(
            "BC4 block payload mismatch: got {} bytes, expected {expected}",
            data.len()
        ));
    }
    let mut decoded = vec![0u8; width as usize * height as usize];
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let offset = ((block_y * blocks_x + block_x) as usize) * BLOCK_BYTES;
            let values = decode_block(
                data[offset..offset + BLOCK_BYTES]
                    .try_into()
                    .expect("BC4 block slice"),
            );
            for local_y in 0..4 {
                for local_x in 0..4 {
                    let x = block_x * 4 + local_x;
                    let y = block_y * 4 + local_y;
                    if x < width && y < height {
                        decoded[(y * width + x) as usize] =
                            values[(local_y * 4 + local_x) as usize];
                    }
                }
            }
        }
    }
    Ok(decoded)
}

fn encode_block(values: &[u8; 16]) -> [u8; BLOCK_BYTES] {
    let endpoint0 = *values.iter().max().expect("fixed-size block");
    let endpoint1 = *values.iter().min().expect("fixed-size block");
    let palette = palette(endpoint0, endpoint1);
    let mut indices = 0u64;
    for (pixel, value) in values.iter().enumerate() {
        let index = palette
            .iter()
            .enumerate()
            .min_by_key(|(_, candidate)| i32::from(**candidate).abs_diff(i32::from(*value)))
            .map(|(index, _)| index as u64)
            .expect("BC4 palette");
        indices |= index << (pixel * 3);
    }
    let mut block = [0u8; BLOCK_BYTES];
    block[0] = endpoint0;
    block[1] = endpoint1;
    for byte in 0..6 {
        block[2 + byte] = ((indices >> (byte * 8)) & 0xff) as u8;
    }
    block
}

fn decode_block(block: &[u8; BLOCK_BYTES]) -> [u8; 16] {
    let palette = palette(block[0], block[1]);
    let mut packed = 0u64;
    for byte in 0..6 {
        packed |= u64::from(block[2 + byte]) << (byte * 8);
    }
    let mut values = [0u8; 16];
    for (pixel, value) in values.iter_mut().enumerate() {
        *value = palette[((packed >> (pixel * 3)) & 7) as usize];
    }
    values
}

fn palette(endpoint0: u8, endpoint1: u8) -> [u8; 8] {
    let mut values = [0u8; 8];
    values[0] = endpoint0;
    values[1] = endpoint1;
    if endpoint0 > endpoint1 {
        for index in 2..8 {
            let a = 8 - index;
            let b = index - 1;
            values[index] =
                ((a * usize::from(endpoint0) + b * usize::from(endpoint1) + 3) / 7) as u8;
        }
    } else {
        for index in 2..6 {
            let a = 6 - index;
            let b = index - 1;
            values[index] =
                ((a * usize::from(endpoint0) + b * usize::from(endpoint1) + 2) / 5) as u8;
        }
        values[6] = 0;
        values[7] = 255;
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_block_round_trips_exactly() {
        let values = [117u8; 16];
        assert_eq!(decode_block(&encode_block(&values)), values);
    }

    #[test]
    fn gradient_error_is_bounded() {
        let values = std::array::from_fn(|index| (index * 17) as u8);
        let decoded = decode_block(&encode_block(&values));
        let max_error = values
            .iter()
            .zip(decoded)
            .map(|(a, b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(max_error <= 18, "max BC4 error was {max_error}");
    }

    #[test]
    fn decodes_spec_constructed_reference_block() {
        // Independently constructed BC4 UNORM block: endpoints 255/0 and all
        // 3-bit selectors zero. This is the canonical solid-white block and
        // catches bit-order/interoperability drift against external decoders.
        let reference = [255, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(decode_block(&reference), [255; 16]);
    }

    #[test]
    fn round_trips_a_spec_constructed_eight_value_reference_block() {
        // A non-degenerate reference block exercising every selector value.
        // endpoint0 = 240 > endpoint1 = 16 selects the 8-value palette
        // [240, 16, 208, 176, 144, 112, 80, 48] (the six interpolants are
        // (a*e0 + b*e1 + 3) / 7 for a = 8-i, b = i-1). The selector bytes are
        // indices 0..7 repeated twice, packed 3 bits per texel little-endian.
        // Both literals were derived from the BC4 UNORM layout, not from this
        // codec, so bit-order drift against an external decoder is caught.
        const REFERENCE: [u8; BLOCK_BYTES] = [240, 16, 136, 198, 250, 136, 198, 250];
        const VALUES: [u8; 16] = [
            240, 16, 208, 176, 144, 112, 80, 48, 240, 16, 208, 176, 144, 112, 80, 48,
        ];

        assert_eq!(decode_block(&REFERENCE), VALUES);
        assert_eq!(encode_block(&VALUES), REFERENCE);
    }
}
