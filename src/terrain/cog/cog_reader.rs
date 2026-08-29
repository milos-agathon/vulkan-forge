//! P3.1-P3.2: COG HeightReader implementation.

use super::cache::CogTileCache;
use super::error::CogError;
use super::ifd_parser::{
    parse_cog_header, CogHeader, COMPRESSION_DEFLATE, COMPRESSION_DEFLATE_ALT, COMPRESSION_F3DZ,
    COMPRESSION_LZW, COMPRESSION_NONE, SAMPLE_FORMAT_FLOAT, SAMPLE_FORMAT_INT, SAMPLE_FORMAT_UINT,
    TIFF_PREDICTOR_HORIZONTAL, TIFF_PREDICTOR_NONE,
};
use super::range_reader::RangeReader;
use crate::terrain::page_table::HeightReader;
use crate::terrain::tiling::TileBounds;
use glam::Vec2;
use std::path::PathBuf;
use std::sync::Arc;

/// COG-based height reader implementing the HeightReader trait.
pub struct CogHeightReader {
    reader: Arc<RangeReader>,
    header: CogHeader,
    cache: Arc<CogTileCache>,
    runtime: tokio::runtime::Handle,
}

impl CogHeightReader {
    /// Create a new COG height reader from a URL.
    pub async fn new(url: &str, cache_size_mb: u32) -> Result<Self, CogError> {
        Self::new_with_cache_options(url, cache_size_mb, None, cache_size_mb).await
    }

    /// Create a new COG height reader with explicit range-cache options.
    pub async fn new_with_cache_options(
        url: &str,
        cache_size_mb: u32,
        cache_dir: Option<PathBuf>,
        range_cache_budget_mb: u32,
    ) -> Result<Self, CogError> {
        let range_cache_budget_bytes = range_cache_budget_mb as u64 * 1024 * 1024;
        let reader = if url.starts_with("file://") {
            let path = url.strip_prefix("file://").unwrap_or(url);
            RangeReader::new_local_with_cache_options(
                path,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )?
        } else {
            RangeReader::new_with_cache_options(
                url,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )
            .await?
        };

        let reader = Arc::new(reader);
        let header = parse_cog_header(&reader).await?;
        let cache = Arc::new(CogTileCache::new(cache_size_mb));

        let runtime = tokio::runtime::Handle::current();

        Ok(Self {
            reader,
            header,
            cache,
            runtime,
        })
    }

    /// Create with an existing tokio runtime handle.
    pub async fn new_with_runtime(
        url: &str,
        cache_size_mb: u32,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, CogError> {
        Self::new_with_runtime_and_cache_options(url, cache_size_mb, runtime, None, cache_size_mb)
            .await
    }

    /// Create with an existing tokio runtime handle and cache options.
    pub async fn new_with_runtime_and_cache_options(
        url: &str,
        cache_size_mb: u32,
        runtime: tokio::runtime::Handle,
        cache_dir: Option<PathBuf>,
        range_cache_budget_mb: u32,
    ) -> Result<Self, CogError> {
        let range_cache_budget_bytes = range_cache_budget_mb as u64 * 1024 * 1024;
        let reader = if url.starts_with("file://") {
            let path = url.strip_prefix("file://").unwrap_or(url);
            RangeReader::new_local_with_cache_options(
                path,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )?
        } else {
            RangeReader::new_with_cache_options(
                url,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )
            .await?
        };

        let reader = Arc::new(reader);
        let header = parse_cog_header(&reader).await?;
        let cache = Arc::new(CogTileCache::new(cache_size_mb));

        Ok(Self {
            reader,
            header,
            cache,
            runtime,
        })
    }

    /// Get geographic bounds (from first IFD).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        if let Some(ifd) = self.header.full_resolution() {
            (0.0, 0.0, ifd.width as f64, ifd.height as f64)
        } else {
            (0.0, 0.0, 1.0, 1.0)
        }
    }

    /// Get number of overview levels.
    pub fn overview_count(&self) -> usize {
        self.header.ifds.len()
    }

    /// Get the COG header for inspection.
    pub fn header(&self) -> &CogHeader {
        &self.header
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> super::cache::CogCacheStats {
        let mut stats = self.cache.stats();
        stats.byte_cache_used_bytes = self.reader.stats().cached_bytes();
        stats.byte_cache_budget_bytes = self.reader.byte_cache_budget_bytes();
        stats.disk_cache_used_bytes = self.reader.stats().disk_cached_bytes();
        stats.disk_cache_budget_bytes = self.reader.disk_cache_budget_bytes();
        stats
    }

    /// Read a specific tile at given LOD.
    pub fn read_tile(&self, tile_x: u32, tile_y: u32, lod: u32) -> Result<Vec<f32>, CogError> {
        let ifd = self.header.select_ifd_for_lod(lod)?;

        let cache_key = (tile_x, tile_y, lod);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let tile_idx = ifd
            .tile_index(tile_x, tile_y)
            .ok_or(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            })?;

        if tile_idx >= ifd.tile_offsets.len() || tile_idx >= ifd.tile_byte_counts.len() {
            return Err(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            });
        }

        let offset = ifd.tile_offsets[tile_idx];
        let byte_count = ifd.tile_byte_counts[tile_idx];

        let reader = self.reader.clone();
        let compression = ifd.compression;
        let bits_per_sample = ifd.bits_per_sample;
        let sample_format = ifd.sample_format;
        let predictor = ifd.predictor;
        let tile_width = ifd.tile_width;
        let tile_height = ifd.tile_height;

        let heights = self.runtime.block_on(async move {
            let compressed = reader.read_range(offset, byte_count).await?;
            if compression == COMPRESSION_F3DZ
                && (bits_per_sample != 32 || sample_format != SAMPLE_FORMAT_FLOAT)
            {
                return Err(CogError::InvalidIfd(
                    "F3DZ TIFF tiles require 32-bit floating-point sample metadata".into(),
                ));
            }
            let decompressed =
                decompress_tile(&compressed, compression, Some((tile_width, tile_height)))?;
            decode_heights(
                &decompressed,
                bits_per_sample,
                sample_format,
                tile_width,
                tile_height,
                if compression == COMPRESSION_F3DZ {
                    TIFF_PREDICTOR_NONE
                } else {
                    predictor
                },
            )
        })?;

        let tile_size = (tile_width as usize)
            .checked_mul(tile_height as usize)
            .ok_or_else(|| CogError::InvalidIfd("tile element count overflow".into()))?;
        let memory_bytes = tile_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CogError::InvalidIfd("tile byte size overflow".into()))?;
        self.cache.insert(cache_key, heights.clone(), memory_bytes);

        Ok(heights)
    }

    /// Read tile async.
    pub async fn read_tile_async(
        &self,
        tile_x: u32,
        tile_y: u32,
        lod: u32,
    ) -> Result<Vec<f32>, CogError> {
        let ifd = self.header.select_ifd_for_lod(lod)?;

        let cache_key = (tile_x, tile_y, lod);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let tile_idx = ifd
            .tile_index(tile_x, tile_y)
            .ok_or(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            })?;

        if tile_idx >= ifd.tile_offsets.len() || tile_idx >= ifd.tile_byte_counts.len() {
            return Err(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            });
        }

        let offset = ifd.tile_offsets[tile_idx];
        let byte_count = ifd.tile_byte_counts[tile_idx];

        let compressed = self.reader.read_range(offset, byte_count).await?;
        if ifd.compression == COMPRESSION_F3DZ
            && (ifd.bits_per_sample != 32 || ifd.sample_format != SAMPLE_FORMAT_FLOAT)
        {
            return Err(CogError::InvalidIfd(
                "F3DZ TIFF tiles require 32-bit floating-point sample metadata".into(),
            ));
        }
        let decompressed = decompress_tile(
            &compressed,
            ifd.compression,
            Some((ifd.tile_width, ifd.tile_height)),
        )?;
        let heights = decode_heights(
            &decompressed,
            ifd.bits_per_sample,
            ifd.sample_format,
            ifd.tile_width,
            ifd.tile_height,
            if ifd.compression == COMPRESSION_F3DZ {
                TIFF_PREDICTOR_NONE
            } else {
                ifd.predictor
            },
        )?;

        let tile_size = (ifd.tile_width as usize)
            .checked_mul(ifd.tile_height as usize)
            .ok_or_else(|| CogError::InvalidIfd("tile element count overflow".into()))?;
        let memory_bytes = tile_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CogError::InvalidIfd("tile byte size overflow".into()))?;
        self.cache.insert(cache_key, heights.clone(), memory_bytes);

        Ok(heights)
    }
}

impl HeightReader for CogHeightReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: crate::terrain::tiling::TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        match self.read_tile(tile_id.x, tile_id.y, tile_id.lod) {
            Ok(heights) => {
                if heights.len() == (width * height) as usize {
                    heights
                } else {
                    resample_tile(&heights, width, height)
                }
            }
            Err(e) => {
                log::warn!("COG tile read failed: {:?}", e);
                vec![0.0f32; (width * height) as usize]
            }
        }
    }
}

fn decompress_tile(
    data: &[u8],
    compression: u16,
    expected_dimensions: Option<(u32, u32)>,
) -> Result<Vec<u8>, CogError> {
    match compression {
        COMPRESSION_NONE => Ok(data.to_vec()),
        COMPRESSION_DEFLATE | COMPRESSION_DEFLATE_ALT => {
            use flate2::read::ZlibDecoder;
            use std::io::Read;

            let mut decoder = ZlibDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| CogError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        COMPRESSION_LZW => decompress_lzw(data),
        COMPRESSION_F3DZ => {
            let decoded = crate::codec::f3dz::decode_dem(data, None)
                .map_err(|error| CogError::DecompressionError(error.to_string()))?;
            if let Some((width, height)) = expected_dimensions {
                if decoded.width != width || decoded.height != height {
                    return Err(CogError::DecompressionError(format!(
                        "F3DZ grid {}x{} does not match TIFF tile {}x{}",
                        decoded.width, decoded.height, width, height
                    )));
                }
            }
            let mut bytes = Vec::with_capacity(decoded.values.len() * 4);
            for value in decoded.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(bytes)
        }
        other => Err(CogError::UnsupportedCompression(other)),
    }
}

fn decompress_lzw(data: &[u8]) -> Result<Vec<u8>, CogError> {
    const CLEAR_CODE: u16 = 256;
    const EOI_CODE: u16 = 257;

    let mut output = Vec::new();
    let mut table: Vec<Vec<u8>> = (0u16..256).map(|i| vec![i as u8]).collect();
    table.push(Vec::new()); // CLEAR_CODE placeholder
    table.push(Vec::new()); // EOI_CODE placeholder

    let mut bit_reader = LzwBitReader::new(data);
    let mut code_size = 9u8;
    let mut prev_code: Option<u16> = None;

    while let Some(code) = bit_reader.read_bits(code_size) {
        if code == EOI_CODE {
            break;
        }

        if code == CLEAR_CODE {
            table.truncate(258);
            code_size = 9;
            prev_code = None;
            continue;
        }

        let entry = if (code as usize) < table.len() {
            table[code as usize].clone()
        } else if code as usize == table.len() {
            if let Some(pc) = prev_code {
                let mut e = table[pc as usize].clone();
                e.push(e[0]);
                e
            } else {
                return Err(CogError::DecompressionError(
                    "LZW: invalid code sequence".into(),
                ));
            }
        } else {
            return Err(CogError::DecompressionError(format!(
                "LZW: code {} out of range (table size {})",
                code,
                table.len()
            )));
        };

        output.extend_from_slice(&entry);

        if let Some(pc) = prev_code {
            if table.len() < 4096 {
                let mut new_entry = table[pc as usize].clone();
                new_entry.push(entry[0]);
                table.push(new_entry);

                // TIFF LZW uses EarlyChange=1: increase the width one entry early.
                if table.len() == (1 << code_size) - 1 && code_size < 12 {
                    code_size += 1;
                }
            }
        }

        prev_code = Some(code);
    }

    Ok(output)
}

struct LzwBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> LzwBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, count: u8) -> Option<u16> {
        let mut result: u32 = 0;
        let mut bits_read = 0u8;

        while bits_read < count {
            if self.byte_pos >= self.data.len() {
                return None;
            }

            let bits_available = 8 - self.bit_pos;
            let bits_needed = count - bits_read;
            let bits_to_read = bits_available.min(bits_needed);

            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let shift = 8 - self.bit_pos - bits_to_read;
            let bits = (self.data[self.byte_pos] >> shift) & mask;

            result = (result << bits_to_read) | (bits as u32);
            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Some(result as u16)
    }
}

fn decode_heights(
    data: &[u8],
    bits_per_sample: u16,
    sample_format: u16,
    tile_width: u32,
    tile_height: u32,
    predictor: u16,
) -> Result<Vec<f32>, CogError> {
    let pixel_count = (tile_width as usize)
        .checked_mul(tile_height as usize)
        .ok_or_else(|| CogError::InvalidIfd("tile element count overflow".into()))?;
    let mut heights = Vec::with_capacity(pixel_count);
    let bytes_per_sample = (bits_per_sample as usize + 7) / 8;
    let data = apply_predictor(data, predictor, bytes_per_sample, tile_width, tile_height)?;
    let data = data.as_slice();

    match (bits_per_sample, sample_format) {
        (32, SAMPLE_FORMAT_FLOAT) => {
            let needed = pixel_count
                .checked_mul(4)
                .ok_or_else(|| CogError::InvalidIfd("f32 tile byte size overflow".into()))?;
            if data.len() < needed {
                return Err(CogError::InvalidIfd(format!(
                    "Data too short: {} < {}",
                    data.len(),
                    needed
                )));
            }
            for i in 0..pixel_count {
                heights.push(f32::from_le_bytes(read_le_bytes4(data, i * 4)));
            }
        }
        (64, SAMPLE_FORMAT_FLOAT) => {
            if data.len()
                < pixel_count
                    .checked_mul(8)
                    .ok_or_else(|| CogError::InvalidIfd("f64 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for f64".into()));
            }
            for i in 0..pixel_count {
                heights.push(f64::from_le_bytes(read_le_bytes8(data, i * 8)) as f32);
            }
        }
        (16, SAMPLE_FORMAT_UINT) => {
            if data.len()
                < pixel_count
                    .checked_mul(2)
                    .ok_or_else(|| CogError::InvalidIfd("u16 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for u16".into()));
            }
            for i in 0..pixel_count {
                let val = u16::from_le_bytes(read_le_bytes2(data, i * 2));
                heights.push(val as f32);
            }
        }
        (16, SAMPLE_FORMAT_INT) => {
            if data.len()
                < pixel_count
                    .checked_mul(2)
                    .ok_or_else(|| CogError::InvalidIfd("i16 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for i16".into()));
            }
            for i in 0..pixel_count {
                let val = i16::from_le_bytes(read_le_bytes2(data, i * 2));
                heights.push(val as f32);
            }
        }
        (32, SAMPLE_FORMAT_INT) => {
            if data.len()
                < pixel_count
                    .checked_mul(4)
                    .ok_or_else(|| CogError::InvalidIfd("i32 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for i32".into()));
            }
            for i in 0..pixel_count {
                let val = i32::from_le_bytes(read_le_bytes4(data, i * 4));
                heights.push(val as f32);
            }
        }
        (8, _) => {
            for &byte in data.iter().take(pixel_count) {
                heights.push(byte as f32);
            }
        }
        _ => {
            return Err(CogError::UnsupportedSampleFormat {
                bits: bits_per_sample,
                format: sample_format,
            });
        }
    }

    while heights.len() < pixel_count {
        heights.push(0.0);
    }

    Ok(heights)
}

fn resample_tile(src: &[f32], dst_width: u32, dst_height: u32) -> Vec<f32> {
    let src_side = (src.len() as f32).sqrt() as u32;
    if src_side == 0 {
        return vec![0.0f32; (dst_width * dst_height) as usize];
    }

    let mut dst = Vec::with_capacity((dst_width * dst_height) as usize);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = (x as f32 / dst_width as f32 * src_side as f32) as u32;
            let src_y = (y as f32 / dst_height as f32 * src_side as f32) as u32;
            let src_idx = (src_y.min(src_side - 1) * src_side + src_x.min(src_side - 1)) as usize;
            dst.push(src.get(src_idx).copied().unwrap_or(0.0));
        }
    }
    dst
}

fn apply_predictor(
    data: &[u8],
    predictor: u16,
    bytes_per_sample: usize,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<u8>, CogError> {
    if predictor == TIFF_PREDICTOR_NONE {
        return Ok(data.to_vec());
    }
    if predictor != TIFF_PREDICTOR_HORIZONTAL {
        return Err(CogError::InvalidIfd(format!(
            "Unsupported TIFF predictor: {}",
            predictor
        )));
    }
    if !matches!(bytes_per_sample, 1 | 2 | 4 | 8) {
        return Err(CogError::InvalidIfd(format!(
            "Unsupported predictor sample width: {}",
            bytes_per_sample
        )));
    }

    let row_bytes = (tile_width as usize)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| CogError::InvalidIfd("predictor row size overflow".into()))?;
    let needed = row_bytes
        .checked_mul(tile_height as usize)
        .ok_or_else(|| CogError::InvalidIfd("predictor payload size overflow".into()))?;
    if data.len() < needed {
        return Err(CogError::InvalidIfd(format!(
            "Data too short for predictor: {} < {}",
            data.len(),
            needed
        )));
    }

    let mut out = data.to_vec();
    for row in 0..tile_height as usize {
        let row_start = row * row_bytes;
        for col in 1..tile_width as usize {
            let prev = row_start + (col - 1) * bytes_per_sample;
            let cur = row_start + col * bytes_per_sample;
            match bytes_per_sample {
                1 => out[cur] = out[cur].wrapping_add(out[prev]),
                2 => {
                    let a = u16::from_le_bytes(read_le_bytes2(&out, prev));
                    let b = u16::from_le_bytes(read_le_bytes2(&out, cur));
                    out[cur..cur + 2].copy_from_slice(&b.wrapping_add(a).to_le_bytes());
                }
                4 => {
                    let a = u32::from_le_bytes(read_le_bytes4(&out, prev));
                    let b = u32::from_le_bytes(read_le_bytes4(&out, cur));
                    out[cur..cur + 4].copy_from_slice(&b.wrapping_add(a).to_le_bytes());
                }
                8 => {
                    let a = u64::from_le_bytes(read_le_bytes8(&out, prev));
                    let b = u64::from_le_bytes(read_le_bytes8(&out, cur));
                    out[cur..cur + 8].copy_from_slice(&b.wrapping_add(a).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(out)
}

fn read_le_bytes2(data: &[u8], offset: usize) -> [u8; 2] {
    [data[offset], data[offset + 1]]
}

fn read_le_bytes4(data: &[u8], offset: usize) -> [u8; 4] {
    [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]
}

fn read_le_bytes8(data: &[u8], offset: usize) -> [u8; 8] {
    [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_msb_codes(codes: &[(u16, u8)]) -> Vec<u8> {
        let mut packed = Vec::new();
        let mut byte = 0u8;
        let mut bits_in_byte = 0u8;

        for &(code, width) in codes {
            for shift in (0..width).rev() {
                byte = (byte << 1) | ((code >> shift) as u8 & 1);
                bits_in_byte += 1;
                if bits_in_byte == 8 {
                    packed.push(byte);
                    byte = 0;
                    bits_in_byte = 0;
                }
            }
        }

        if bits_in_byte != 0 {
            packed.push(byte << (8 - bits_in_byte));
        }
        packed
    }

    #[test]
    fn lzw_tiff_early_change_crosses_9_to_10_bits() {
        const CLEAR_CODE: u16 = 256;
        const EOI_CODE: u16 = 257;
        const FIRST_10_BIT_LITERAL: u16 = 254;

        let mut codes = vec![(CLEAR_CODE, 9)];
        for value in 0u16..300 {
            let width = if value < FIRST_10_BIT_LITERAL { 9 } else { 10 };
            codes.push((value % 256, width));
        }
        codes.push((EOI_CODE, 10));

        let encoded = pack_msb_codes(&codes);
        let expected = (0u16..300)
            .map(|value| (value % 256) as u8)
            .collect::<Vec<_>>();

        assert_eq!(decompress_lzw(&encoded).unwrap(), expected);
    }

    #[test]
    fn decode_heights_applies_horizontal_predictor_to_u16_rows() {
        let encoded: Vec<u8> = [10u16, 2, 3, 20, 4, 5]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();

        let decoded = decode_heights(
            &encoded,
            16,
            SAMPLE_FORMAT_UINT,
            3,
            2,
            TIFF_PREDICTOR_HORIZONTAL,
        )
        .unwrap();

        assert_eq!(decoded, vec![10.0, 12.0, 15.0, 20.0, 24.0, 29.0]);
    }

    #[test]
    fn decode_heights_rejects_short_f32_payload_without_panic() {
        let err = decode_heights(
            &[0, 0, 128],
            32,
            SAMPLE_FORMAT_FLOAT,
            1,
            1,
            TIFF_PREDICTOR_NONE,
        )
        .unwrap_err();

        assert!(matches!(err, CogError::InvalidIfd(message) if message.contains("Data too short")));
    }

    #[test]
    fn decode_heights_rejects_short_f64_payload_without_panic() {
        let err = decode_heights(
            &[0, 0, 0, 0, 0, 0, 0],
            64,
            SAMPLE_FORMAT_FLOAT,
            1,
            1,
            TIFF_PREDICTOR_NONE,
        )
        .unwrap_err();

        assert!(
            matches!(err, CogError::InvalidIfd(message) if message.contains("Data too short for f64"))
        );
    }

    #[test]
    fn predictor_rejects_short_horizontal_payload_without_panic() {
        let err = apply_predictor(&[0, 1, 2], TIFF_PREDICTOR_HORIZONTAL, 2, 2, 1).unwrap_err();

        assert!(matches!(err, CogError::InvalidIfd(message) if message.contains("predictor")));
    }

    #[test]
    fn private_f3dz_compression_branch_decodes_f32_tile_bytes() {
        let source = vec![10.0f32, 10.1, f32::NAN, 10.3];
        let stream = crate::codec::f3dz::encode_dem(
            &source,
            2,
            2,
            &crate::codec::f3dz::EncodeOptions::new(0.05),
        )
        .unwrap();
        let bytes = decompress_tile(&stream, COMPRESSION_F3DZ, Some((2, 2))).unwrap();
        let decoded =
            decode_heights(&bytes, 32, SAMPLE_FORMAT_FLOAT, 2, 2, TIFF_PREDICTOR_NONE).unwrap();
        assert_eq!(decoded.len(), source.len());
        assert!(decoded[2].is_nan());
        assert!(decoded
            .iter()
            .zip(source)
            .filter(|(_, source)| !source.is_nan())
            .all(|(decoded, source)| (*decoded - source).abs() <= 0.05));

        assert!(matches!(
            decompress_tile(&stream, COMPRESSION_F3DZ, Some((4, 1))),
            Err(CogError::DecompressionError(message)) if message.contains("does not match")
        ));
    }
}
