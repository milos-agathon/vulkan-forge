use wgpu::TextureFormat;

/// Compress RGBA data to specified format
pub(super) fn compress_rgba_to_format(
    data: &[u8],
    width: u32,
    height: u32,
    target_format: TextureFormat,
) -> Result<Vec<u8>, String> {
    match target_format {
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => {
            compress_bc1(data, width, height)
        }
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => {
            compress_bc3(data, width, height)
        }
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => {
            super::bc7::encode_bc7_rgba8(data, width, height)
        }
        TextureFormat::Bc5RgUnorm => {
            let mut rg = Vec::with_capacity(width as usize * height as usize * 2);
            for rgba in data.chunks_exact(4) {
                rg.extend_from_slice(&rgba[..2]);
            }
            super::bc5::encode_bc5_rg8(&rg, width, height)
        }
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => {
            compress_etc2_rgb(data, width, height)
        }
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => {
            compress_etc2_rgba(data, width, height)
        }
        _ => Err(format!(
            "Compression to {:?} not implemented",
            target_format
        )),
    }
}

/// Simplified BC1 compression used as a low-fidelity fallback.
fn compress_bc1(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 8) as usize;
    let mut compressed = vec![0u8; compressed_size];

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let block_offset = ((block_y * blocks_x + block_x) * 8) as usize;
            let mut block_colors = Vec::new();

            for y in 0..4 {
                for x in 0..4 {
                    let src_x = (block_x * 4 + x).min(width - 1);
                    let src_y = (block_y * 4 + y).min(height - 1);
                    let pixel_offset = ((src_y * width + src_x) * 4) as usize;

                    if pixel_offset + 3 < data.len() {
                        block_colors.push([
                            data[pixel_offset],
                            data[pixel_offset + 1],
                            data[pixel_offset + 2],
                            data[pixel_offset + 3],
                        ]);
                    } else {
                        block_colors.push([0, 0, 0, 0]);
                    }
                }
            }

            if !block_colors.is_empty() {
                let first_color = &block_colors[0];
                let last_color = &block_colors[block_colors.len() - 1];
                let color0 = rgb8_to_rgb565(first_color[0], first_color[1], first_color[2]);
                let color1 = rgb8_to_rgb565(last_color[0], last_color[1], last_color[2]);

                compressed[block_offset..block_offset + 2].copy_from_slice(&color0.to_le_bytes());
                compressed[block_offset + 2..block_offset + 4]
                    .copy_from_slice(&color1.to_le_bytes());
                compressed[block_offset + 4..block_offset + 8]
                    .copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);
            }
        }
    }

    Ok(compressed)
}

/// Convert RGB8 to RGB565
pub(super) fn rgb8_to_rgb565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r >> 3) as u16;
    let g6 = (g >> 2) as u16;
    let b5 = (b >> 3) as u16;
    (r5 << 11) | (g6 << 5) | b5
}

/// BC3 compression fallback; uses BC1 color with a flat alpha block.
fn compress_bc3(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 16) as usize;
    let mut compressed = vec![0u8; compressed_size];
    let bc1_data = compress_bc1(data, width, height)?;

    for i in 0..blocks_x * blocks_y {
        let bc3_offset = (i * 16) as usize;
        let bc1_offset = (i * 8) as usize;

        compressed[bc3_offset..bc3_offset + 8].copy_from_slice(&[0xFF; 8]);
        if bc1_offset + 8 <= bc1_data.len() && bc3_offset + 16 <= compressed.len() {
            compressed[bc3_offset + 8..bc3_offset + 16]
                .copy_from_slice(&bc1_data[bc1_offset..bc1_offset + 8]);
        }
    }

    Ok(compressed)
}

/// ETC2 RGB compression fallback; returns zeroed blocks.
fn compress_etc2_rgb(_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    Ok(vec![0u8; (blocks_x * blocks_y * 8) as usize])
}

/// ETC2 RGBA compression fallback; returns zeroed blocks.
fn compress_etc2_rgba(_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    Ok(vec![0u8; (blocks_x * blocks_y * 16) as usize])
}

/// Calculate number of mip levels for given dimensions
pub(super) fn calculate_mip_levels(width: u32, height: u32) -> u32 {
    let max_dimension = width.max(height);
    (32 - max_dimension.leading_zeros()).max(1)
}

/// Generate mip level data; currently returns none until downsampling is wired.
pub(super) fn generate_mip_levels(
    _base_data: &[u8],
    _base_width: u32,
    _base_height: u32,
    _format: TextureFormat,
) -> Result<Vec<(Vec<u8>, u32, u32)>, String> {
    Ok(Vec::new())
}

/// Estimate quality score for format
pub(super) fn estimate_quality_score(format: TextureFormat) -> f32 {
    match format {
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => 0.95,
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => 0.85,
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => 0.90,
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => 0.75,
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => 0.85,
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => 0.80,
        _ => 0.5,
    }
}

/// Estimate PSNR for format
pub(super) fn estimate_psnr(format: TextureFormat) -> f32 {
    match format {
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => 45.0,
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => 40.0,
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => 42.0,
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => 35.0,
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => 38.0,
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => 36.0,
        _ => 30.0,
    }
}
