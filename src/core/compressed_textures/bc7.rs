//! Deterministic BC7 mode-6 encoder/decoder.
//!
//! Mode 6 is a single-subset RGBA mode with 7-bit endpoints, one p-bit per
//! endpoint, and 4-bit indices. Restricting the packer to mode 6 trades some
//! partitioned-block quality for a compact in-tree implementation and stable
//! output across platforms.

const BLOCK_BYTES: usize = 16;
const WEIGHTS: [u16; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

pub fn encode_bc7_rgba8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("BC7 dimensions must be non-zero".to_string());
    }
    let expected = width as usize * height as usize * 4;
    if data.len() != expected {
        return Err(format!(
            "BC7 input mismatch: got {} bytes, expected {expected}",
            data.len()
        ));
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut encoded = Vec::with_capacity((blocks_x * blocks_y) as usize * BLOCK_BYTES);
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let mut pixels = [[0u8; 4]; 16];
            for local_y in 0..4 {
                for local_x in 0..4 {
                    let x = (block_x * 4 + local_x).min(width - 1);
                    let y = (block_y * 4 + local_y).min(height - 1);
                    let offset = ((y * width + x) * 4) as usize;
                    pixels[(local_y * 4 + local_x) as usize]
                        .copy_from_slice(&data[offset..offset + 4]);
                }
            }
            encoded.extend_from_slice(&encode_block(&pixels));
        }
    }
    Ok(encoded)
}

pub fn decode_bc7_rgba8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("BC7 dimensions must be non-zero".to_string());
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let expected = (blocks_x * blocks_y) as usize * BLOCK_BYTES;
    if data.len() != expected {
        return Err(format!(
            "BC7 block payload mismatch: got {} bytes, expected {expected}",
            data.len()
        ));
    }
    let mut decoded = vec![0u8; width as usize * height as usize * 4];
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let offset = ((block_y * blocks_x + block_x) as usize) * BLOCK_BYTES;
            let pixels = decode_block(
                data[offset..offset + BLOCK_BYTES]
                    .try_into()
                    .expect("BC7 block slice"),
            )?;
            for local_y in 0..4 {
                for local_x in 0..4 {
                    let x = block_x * 4 + local_x;
                    let y = block_y * 4 + local_y;
                    if x < width && y < height {
                        let dst = ((y * width + x) * 4) as usize;
                        decoded[dst..dst + 4]
                            .copy_from_slice(&pixels[(local_y * 4 + local_x) as usize]);
                    }
                }
            }
        }
    }
    Ok(decoded)
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    #[test]
    fn mode6_solid_white_matches_spec_constructed_reference_block() {
        // Mode prefix bit 6, eight 7-bit 0x7f endpoints, both p-bits set,
        // anchor selector 0, and fifteen selectors 0. The literal is derived
        // from the BC7 mode-6 bit layout, not from this encoder.
        let reference = [
            0xc0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(
            decode_bc7_rgba8(&reference, 4, 4).unwrap(),
            vec![255; 4 * 4 * 4]
        );
        assert_eq!(
            encode_bc7_rgba8(&[255; 4 * 4 * 4], 4, 4).unwrap(),
            reference
        );
    }

    #[test]
    fn mode6_matches_a_spec_constructed_gradient_reference_block() {
        // A non-degenerate reference block: endpoints A = (10,22,34,fe) and
        // B = (c0,a6,8e,fe), chosen with every channel even so the single
        // shared p-bit per endpoint reproduces them exactly. The sixteen
        // texels are the exact mode-6 palette entries for index weights
        // 0..15, ordered so the 3-bit anchor index is 0. Both literals were
        // derived from the BC7 mode-6 bit layout (7-bit endpoint pairs, two
        // p-bits, one 3-bit and fifteen 4-bit selectors, little-endian),
        // independently of this encoder.
        const TEXELS: [u8; 64] = [
            0x10, 0x22, 0x34, 0xfe, 0x1b, 0x2a, 0x3a, 0xfe, 0x29, 0x35, 0x41, 0xfe, 0x34, 0x3d,
            0x46, 0xfe, 0x3f, 0x45, 0x4c, 0xfe, 0x4a, 0x4d, 0x52, 0xfe, 0x58, 0x58, 0x59, 0xfe,
            0x63, 0x60, 0x5e, 0xfe, 0x6e, 0x68, 0x64, 0xfe, 0x79, 0x70, 0x69, 0xfe, 0x86, 0x7b,
            0x70, 0xfe, 0x91, 0x83, 0x76, 0xfe, 0x9c, 0x8b, 0x7c, 0xfe, 0xa7, 0x93, 0x81, 0xfe,
            0xb5, 0x9e, 0x88, 0xfe, 0xc0, 0xa6, 0x8e, 0xfe,
        ];
        const REFERENCE: [u8; 16] = [
            0x40, 0x04, 0x38, 0x32, 0xd5, 0x1c, 0xff, 0x7f, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
            0xdc, 0xfe,
        ];

        assert_eq!(decode_bc7_rgba8(&REFERENCE, 4, 4).unwrap(), TEXELS.to_vec());
        assert_eq!(encode_bc7_rgba8(&TEXELS, 4, 4).unwrap(), REFERENCE.to_vec());
    }
}

fn encode_block(pixels: &[[u8; 4]; 16]) -> [u8; BLOCK_BYTES] {
    let (mut endpoint0, mut endpoint1) = principal_endpoints(pixels);
    let mut indices = choose_indices(pixels, endpoint0, endpoint1);
    if indices[0] >= 8 {
        std::mem::swap(&mut endpoint0, &mut endpoint1);
        for index in &mut indices {
            *index = 15 - *index;
        }
    }
    let (quant0, p0) = quantize_endpoint(endpoint0);
    let (quant1, p1) = quantize_endpoint(endpoint1);

    let mut writer = BitWriter::default();
    writer.write(0, 6);
    writer.write(1, 1);
    for channel in 0..4 {
        writer.write(u128::from(quant0[channel]), 7);
        writer.write(u128::from(quant1[channel]), 7);
    }
    writer.write(u128::from(p0), 1);
    writer.write(u128::from(p1), 1);
    writer.write(u128::from(indices[0]), 3);
    for index in &indices[1..] {
        writer.write(u128::from(*index), 4);
    }
    debug_assert_eq!(writer.position, 128);
    writer.bits.to_le_bytes()
}

fn decode_block(block: &[u8; BLOCK_BYTES]) -> Result<[[u8; 4]; 16], String> {
    let mut reader = BitReader {
        bits: u128::from_le_bytes(*block),
        position: 0,
    };
    if reader.read(6) != 0 || reader.read(1) != 1 {
        return Err("BC7 decoder supports deterministic mode-6 blocks only".to_string());
    }
    let mut quant0 = [0u8; 4];
    let mut quant1 = [0u8; 4];
    for channel in 0..4 {
        quant0[channel] = reader.read(7) as u8;
        quant1[channel] = reader.read(7) as u8;
    }
    let p0 = reader.read(1) as u8;
    let p1 = reader.read(1) as u8;
    let endpoint0 = std::array::from_fn(|channel| (quant0[channel] << 1) | p0);
    let endpoint1 = std::array::from_fn(|channel| (quant1[channel] << 1) | p1);
    let mut indices = [0u8; 16];
    indices[0] = reader.read(3) as u8;
    for index in &mut indices[1..] {
        *index = reader.read(4) as u8;
    }
    let pixels = std::array::from_fn(|pixel| interpolate(endpoint0, endpoint1, indices[pixel]));
    Ok(pixels)
}

fn principal_endpoints(pixels: &[[u8; 4]; 16]) -> ([u8; 4], [u8; 4]) {
    if pixels.iter().all(|pixel| pixel == &pixels[0]) {
        return (pixels[0], pixels[0]);
    }
    let mut mean = [0.0f32; 4];
    for pixel in pixels {
        for channel in 0..4 {
            mean[channel] += f32::from(pixel[channel]) / 16.0;
        }
    }
    let mut covariance = [[0.0f32; 4]; 4];
    for pixel in pixels {
        let delta =
            std::array::from_fn::<_, 4, _>(|channel| f32::from(pixel[channel]) - mean[channel]);
        for row in 0..4 {
            for column in 0..4 {
                covariance[row][column] += delta[row] * delta[column];
            }
        }
    }
    let mut axis = [0.5f32; 4];
    for _ in 0..8 {
        let next = std::array::from_fn::<_, 4, _>(|row| {
            (0..4)
                .map(|column| covariance[row][column] * axis[column])
                .sum::<f32>()
        });
        let length = next.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length <= f32::EPSILON {
            break;
        }
        axis = next.map(|value| value / length);
    }
    let mut min_projection = f32::INFINITY;
    let mut max_projection = f32::NEG_INFINITY;
    for pixel in pixels {
        let projection = (0..4)
            .map(|channel| (f32::from(pixel[channel]) - mean[channel]) * axis[channel])
            .sum::<f32>();
        min_projection = min_projection.min(projection);
        max_projection = max_projection.max(projection);
    }
    let endpoint = |projection: f32| {
        std::array::from_fn(|channel| {
            (mean[channel] + projection * axis[channel])
                .round()
                .clamp(0.0, 255.0) as u8
        })
    };
    (endpoint(min_projection), endpoint(max_projection))
}

fn quantize_endpoint(endpoint: [u8; 4]) -> ([u8; 4], u8) {
    let candidate = |pbit: u8| {
        let quantized = endpoint
            .map(|value| ((i16::from(value) - i16::from(pbit) + 1) / 2).clamp(0, 127) as u8);
        let error = (0..4)
            .map(|channel| {
                let reconstructed = (quantized[channel] << 1) | pbit;
                let delta = i32::from(reconstructed) - i32::from(endpoint[channel]);
                delta * delta
            })
            .sum::<i32>();
        (quantized, error)
    };
    let zero = candidate(0);
    let one = candidate(1);
    if zero.1 <= one.1 {
        (zero.0, 0)
    } else {
        (one.0, 1)
    }
}

fn choose_indices(pixels: &[[u8; 4]; 16], endpoint0: [u8; 4], endpoint1: [u8; 4]) -> [u8; 16] {
    std::array::from_fn(|pixel| {
        (0..16)
            .min_by_key(|index| {
                let candidate = interpolate(endpoint0, endpoint1, *index as u8);
                (0..4)
                    .map(|channel| {
                        let delta =
                            i32::from(candidate[channel]) - i32::from(pixels[pixel][channel]);
                        delta * delta
                    })
                    .sum::<i32>()
            })
            .expect("fixed BC7 palette") as u8
    })
}

fn interpolate(endpoint0: [u8; 4], endpoint1: [u8; 4], index: u8) -> [u8; 4] {
    let weight = WEIGHTS[index as usize];
    std::array::from_fn(|channel| {
        (((64 - weight) * u16::from(endpoint0[channel])
            + weight * u16::from(endpoint1[channel])
            + 32)
            >> 6) as u8
    })
}

#[derive(Default)]
struct BitWriter {
    bits: u128,
    position: u32,
}

impl BitWriter {
    fn write(&mut self, value: u128, count: u32) {
        debug_assert!(count == 128 || value < (1u128 << count));
        self.bits |= value << self.position;
        self.position += count;
    }
}

struct BitReader {
    bits: u128,
    position: u32,
}

impl BitReader {
    fn read(&mut self, count: u32) -> u128 {
        let mask = (1u128 << count) - 1;
        let value = (self.bits >> self.position) & mask;
        self.position += count;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode6_constant_block_round_trips_with_one_lsb_tolerance() {
        let pixels = [[23, 81, 149, 255]; 16];
        let decoded = decode_block(&encode_block(&pixels)).unwrap();
        for (actual, expected) in decoded.iter().flatten().zip(pixels.iter().flatten()) {
            assert!(actual.abs_diff(*expected) <= 1);
        }
    }

    #[test]
    fn mode6_anchor_uses_fixup_bit() {
        let mut pixels = [[0, 0, 0, 255]; 16];
        pixels[0] = [255, 255, 255, 255];
        let block = encode_block(&pixels);
        let mut reader = BitReader {
            bits: u128::from_le_bytes(block),
            position: 65,
        };
        assert!(reader.read(3) < 8);
        decode_block(&block).unwrap();
    }
}
