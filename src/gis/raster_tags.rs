//! GeoTIFF tag and GeoKey readers: per-band dtype/nodata metadata,
//! compression names, the affine transform, and CRS extraction (EPSG GeoKeys
//! and embedded WKT). Pure metadata — no pixel data is decoded here.

use std::collections::HashMap;

use tiff::decoder::Decoder;
use tiff::tags::Tag;

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::{AffineTransform, RasterDType};

const FORGE3D_NODATA_PREFIX: &str = "forge3d:nodata_per_band=";

pub(crate) fn sample_count<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<u16> {
    if let Some(samples) = decoder.find_tag_unsigned::<u16>(Tag::SamplesPerPixel)? {
        return Ok(samples);
    }
    Ok(1)
}

pub(crate) fn dtype_per_band<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    band_count: u16,
) -> GisResult<Vec<String>> {
    let bits = decoder
        .find_tag_unsigned_vec::<u16>(Tag::BitsPerSample)?
        .unwrap_or_else(|| vec![8; band_count as usize]);
    let sample_formats = decoder
        .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)?
        .unwrap_or_else(|| vec![1; band_count as usize]);

    let mut dtypes = Vec::with_capacity(band_count as usize);
    for index in 0..band_count as usize {
        let bit_depth = bits.get(index).copied().unwrap_or_else(|| bits[0]);
        let sample_format = sample_formats.get(index).copied().unwrap_or(1);
        let dtype = dtype_from_tags(bit_depth, sample_format)?;
        dtypes.push(dtype.name().to_string());
    }
    Ok(dtypes)
}

fn dtype_from_tags(bits: u16, sample_format: u16) -> GisResult<RasterDType> {
    match (bits, sample_format) {
        (8, 1) => Ok(RasterDType::UInt8),
        (16, 2) => Ok(RasterDType::Int16),
        (16, 1) => Ok(RasterDType::UInt16),
        (32, 2) => Ok(RasterDType::Int32),
        (32, 1) => Ok(RasterDType::UInt32),
        (32, 3) => Ok(RasterDType::Float32),
        (64, 3) => Ok(RasterDType::Float64),
        _ => Err(GisError::UnsupportedDType(format!(
            "unsupported TIFF dtype bits_per_sample={bits}, sample_format={sample_format}"
        ))),
    }
}

pub(crate) fn nodata_per_band<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    band_count: u16,
) -> GisResult<Vec<Option<f64>>> {
    if let Some(value) = decoder.find_tag(Tag::ImageDescription)? {
        let description = value
            .into_string()
            .map_err(|err| GisError::InvalidRaster(err.to_string()))?;
        if let Some(values) = parse_forge3d_nodata(&description, band_count)? {
            return Ok(values);
        }
    }

    match decoder.find_tag(Tag::GdalNodata)? {
        Some(value) => {
            let raw = value
                .into_string()
                .map_err(|err| GisError::InvalidRaster(err.to_string()))?;
            let parsed = parse_nodata_string(&raw)?;
            Ok(vec![parsed; band_count as usize])
        }
        None => Ok(vec![None; band_count as usize]),
    }
}

fn parse_forge3d_nodata(raw: &str, band_count: u16) -> GisResult<Option<Vec<Option<f64>>>> {
    let Some(rest) = raw
        .trim_matches(char::from(0))
        .strip_prefix(FORGE3D_NODATA_PREFIX)
    else {
        return Ok(None);
    };
    let values = rest
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.eq_ignore_ascii_case("none") || part.is_empty() {
                Ok(None)
            } else if part.eq_ignore_ascii_case("nan") {
                Ok(Some(f64::NAN))
            } else {
                part.parse::<f64>().map(Some).map_err(|_| {
                    GisError::InvalidRaster(format!(
                        "invalid forge3d per-band nodata value: {part:?}"
                    ))
                })
            }
        })
        .collect::<GisResult<Vec<_>>>()?;
    if values.len() != band_count as usize {
        return Err(GisError::InvalidRaster(format!(
            "forge3d per-band nodata count {} does not match band count {band_count}",
            values.len()
        )));
    }
    Ok(Some(values))
}

fn parse_nodata_string(raw: &str) -> GisResult<Option<f64>> {
    let trimmed = raw.trim_matches(char::from(0)).trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.eq_ignore_ascii_case("nan") {
        return Ok(Some(f64::NAN));
    }
    trimmed
        .parse::<f64>()
        .map(Some)
        .map_err(|_| GisError::InvalidRaster(format!("invalid GDAL_NODATA tag value: {trimmed:?}")))
}

pub(crate) fn compression_name<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<Option<String>> {
    Ok(decoder
        .find_tag_unsigned::<u16>(Tag::Compression)?
        .map(|value| match value {
            1 => "NONE".to_string(),
            5 => "LZW".to_string(),
            8 | 32946 => "DEFLATE".to_string(),
            32773 => "PACKBITS".to_string(),
            other => format!("UNKNOWN({other})"),
        }))
}

pub(crate) fn read_transform<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<Option<AffineTransform>> {
    if let Some(values) = decoder.find_tag(Tag::ModelTransformationTag)? {
        let values = values
            .into_f64_vec()
            .map_err(|err| GisError::InvalidTransform(err.to_string()))?;
        if values.len() != 16 {
            return Err(GisError::InvalidTransform(format!(
                "ModelTransformationTag must contain exactly 16 doubles, found {}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GisError::InvalidTransform(
                "ModelTransformationTag coefficients must be finite".to_string(),
            ));
        }
        if values[2] != 0.0
            || values[6] != 0.0
            || values[8] != 0.0
            || values[9] != 0.0
            || values[12] != 0.0
            || values[13] != 0.0
            || values[14] != 0.0
            || values[15] != 1.0
        {
            return Err(GisError::InvalidTransform(
                "ModelTransformationTag is not a finite 2D affine transform".to_string(),
            ));
        }
        let transform = AffineTransform::new([
            values[0], values[1], values[3], values[4], values[5], values[7],
        ])?;
        return Ok(Some(transform));
    }

    // Read both tags before deciding absence. A single tag is malformed
    // metadata, not the legacy local-coordinate fallback.
    let pixel_scale = decoder
        .find_tag(Tag::ModelPixelScaleTag)?
        .map(|value| {
            value
                .into_f64_vec()
                .map_err(|err| GisError::InvalidTransform(err.to_string()))
        })
        .transpose()?;
    let tiepoints = decoder
        .find_tag(Tag::ModelTiepointTag)?
        .map(|value| {
            value
                .into_f64_vec()
                .map_err(|err| GisError::InvalidTransform(err.to_string()))
        })
        .transpose()?;
    let (pixel_scale, tiepoints) = match (pixel_scale, tiepoints) {
        (None, None) => return Ok(None),
        (Some(_), None) => {
            return Err(GisError::InvalidTransform(
                "ModelPixelScaleTag is present but ModelTiepointTag is missing".to_string(),
            ))
        }
        (None, Some(_)) => {
            return Err(GisError::InvalidTransform(
                "ModelTiepointTag is present but ModelPixelScaleTag is missing".to_string(),
            ))
        }
        (Some(pixel_scale), Some(tiepoints)) => (pixel_scale, tiepoints),
    };
    // The viewer supports exactly one 2D scale/tiepoint transform. Accepting
    // truncated or additional records would silently discard metadata and
    // make the selected placement ambiguous.
    if pixel_scale.len() != 3 || tiepoints.len() != 6 {
        return Err(GisError::InvalidTransform(
            format!(
                "GeoTIFF pixel scale/tiepoint tags must contain exactly 3 and 6 doubles, found {} and {}",
                pixel_scale.len(),
                tiepoints.len()
            ),
        ));
    }
    let scale_x = pixel_scale[0];
    let scale_y = pixel_scale[1];
    let raster_i = tiepoints[0];
    let raster_j = tiepoints[1];
    let model_x = tiepoints[3];
    let model_y = tiepoints[4];
    if pixel_scale.iter().any(|value| !value.is_finite())
        || tiepoints.iter().any(|value| !value.is_finite())
    {
        return Err(GisError::InvalidTransform(
            "GeoTIFF pixel scale/tiepoint coefficients must be finite".to_string(),
        ));
    }
    if scale_x <= 0.0 || scale_y <= 0.0 {
        return Err(GisError::InvalidTransform(
            "GeoTIFF pixel scales must be positive (no mirrored or south-up transforms)"
                .to_string(),
        ));
    }
    let transform = AffineTransform::new([
        scale_x,
        0.0,
        model_x - raster_i * scale_x,
        0.0,
        -scale_y,
        model_y + raster_j * scale_y,
    ])?;
    Ok(Some(transform))
}

pub(crate) fn read_crs<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<Option<(Option<String>, Option<HashMap<String, String>>)>> {
    let geo_ascii = decoder
        .find_tag(Tag::GeoAsciiParamsTag)?
        .map(|value| {
            value
                .into_string()
                .map_err(|err| GisError::InvalidCrs(err.to_string()))
        })
        .transpose()?;
    let directory = match decoder.find_tag(Tag::GeoKeyDirectoryTag)? {
        Some(value) => value
            .into_u16_vec()
            .map_err(|err| GisError::InvalidCrs(err.to_string()))?,
        None => {
            return Ok(match geo_ascii.as_deref() {
                Some(ascii) => {
                    let authority = iau_authority_from_ascii(ascii)?;
                    let wkt = validated_wkt_from_ascii(ascii)?;
                    if authority.is_some() || wkt.is_some() {
                        Some((wkt, authority))
                    } else {
                        None
                    }
                }
                None => None,
            })
        }
    };
    if directory.len() < 4 {
        return Err(GisError::InvalidCrs(
            "GeoKeyDirectoryTag is too short".to_string(),
        ));
    }
    let entry_count = directory[3] as usize;
    if directory.len() < 4 + entry_count * 4 {
        return Err(GisError::InvalidCrs(
            "GeoKeyDirectoryTag entry count exceeds tag length".to_string(),
        ));
    }

    let mut geographic_epsg = None;
    let mut projected_epsg = None;
    let mut iau_geographic_authority = None;
    let mut iau_projected_authority = None;
    let mut wkt = match geo_ascii.as_deref() {
        Some(ascii) => validated_wkt_from_ascii(ascii)?,
        None => None,
    };
    for entry in directory[4..4 + entry_count * 4].chunks_exact(4) {
        let key = entry[0];
        let tag_location = entry[1];
        let count = entry[2];
        let value_offset = entry[3];
        if tag_location == Tag::GeoAsciiParamsTag.to_u16() {
            if let Some(ascii) = &geo_ascii {
                if matches!(key, 1026 | 2049 | 3073) {
                    if let Some(value) = extract_ascii_range(ascii, value_offset, count) {
                        if looks_like_wkt(&value) {
                            wkt = Some(validate_wkt_literal(&value)?);
                        }
                        if let Some(authority) = iau_authority_from_ascii(&value)? {
                            if key == 2049 {
                                iau_geographic_authority = Some(authority);
                            } else {
                                iau_projected_authority = Some(authority);
                            }
                        }
                    }
                }
            }
            continue;
        }
        if tag_location != 0 || count != 1 {
            continue;
        }
        match key {
            2048 if value_offset != 0 && value_offset != 32767 => {
                geographic_epsg = Some(value_offset)
            }
            3072 if value_offset != 0 && value_offset != 32767 => {
                projected_epsg = Some(value_offset)
            }
            _ => {}
        }
    }

    let epsg = projected_epsg.or(geographic_epsg);
    if let Some(code) = epsg {
        // Metadata inspection does not imply reprojection support. Preserve any
        // finite GeoKey authority code verbatim; operations that transform
        // coordinates validate their supported CRS pairs at their own boundary.
        let mut authority = HashMap::new();
        authority.insert("name".to_string(), "EPSG".to_string());
        authority.insert("code".to_string(), code.to_string());
        return Ok(Some((wkt, Some(authority))));
    }
    if let Some(authority) = iau_projected_authority.or(iau_geographic_authority) {
        return Ok(Some((wkt, Some(authority))));
    }
    if let Some(authority) = geo_ascii
        .as_deref()
        .map(iau_authority_from_ascii)
        .transpose()?
        .flatten()
    {
        return Ok(Some((wkt, Some(authority))));
    }

    Ok(wkt.map(|wkt| (Some(wkt), None)))
}

fn iau_authority_from_ascii(ascii: &str) -> GisResult<Option<HashMap<String, String>>> {
    let cleaned = ascii.trim_matches(char::from(0)).trim();
    let segments = cleaned
        .split('|')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    for segment in &segments {
        if let Some((authority, code)) = segment.split_once(':') {
            if authority.eq_ignore_ascii_case("IAU")
                && !code.is_empty()
                && code.chars().all(|ch| ch.is_ascii_digit())
            {
                let mut result = HashMap::new();
                result.insert("name".to_string(), "IAU".to_string());
                result.insert("code".to_string(), code.to_string());
                return Ok(Some(result));
            }
        }
    }
    let prefix = segments
        .into_iter()
        .find(|segment| {
            let upper = segment.to_ascii_uppercase();
            (upper.contains("MOON (2015)") || upper.contains("MARS (2015)"))
                && (upper.contains("/ OCENTRIC") || upper.contains("/ OGRAPHIC"))
        })
        .unwrap_or_default();
    let Some((authority, code)) = prefix.split_once(':') else {
        let upper = prefix.to_ascii_uppercase();
        let mars = upper.contains("MARS (2015)");
        let moon = upper.contains("MOON (2015)");
        if !mars && !moon {
            return Ok(None);
        }
        if upper.contains("/ OGRAPHIC") {
            let code = if upper.contains("EQUIRECTANGULAR, CLON = 180") {
                49916
            } else if upper.contains("EQUIRECTANGULAR") {
                49911
            } else if upper.contains("NORTH POLAR") {
                49931
            } else if upper.contains("SOUTH POLAR") {
                49936
            } else if upper.starts_with("GCS NAME =") {
                49901
            } else {
                return Err(GisError::InvalidCrs(format!(
                    "unsupported IAU planetary citation {prefix:?}"
                )));
            };
            let mut result = HashMap::new();
            result.insert("name".to_string(), "IAU".to_string());
            result.insert("code".to_string(), code.to_string());
            return Ok(Some(result));
        }
        if !upper.contains("/ OCENTRIC") {
            return Ok(None);
        }
        let sphere = upper.contains(" - SPHERE");
        let code = if moon {
            if upper.contains("EQUIRECTANGULAR, CLON = 180") {
                30115
            } else if upper.contains("EQUIRECTANGULAR") {
                30110
            } else if upper.contains("NORTH POLAR") {
                30130
            } else if upper.contains("SOUTH POLAR") {
                30135
            } else if upper.starts_with("GCS NAME =") {
                30100
            } else {
                return Err(GisError::InvalidCrs(format!(
                    "unsupported IAU planetary citation {prefix:?}"
                )));
            }
        } else if mars {
            if upper.contains("EQUIRECTANGULAR, CLON = 180") {
                if sphere {
                    49915
                } else {
                    49917
                }
            } else if upper.contains("EQUIRECTANGULAR") {
                if sphere {
                    49910
                } else {
                    49912
                }
            } else if upper.contains("NORTH POLAR") {
                if sphere {
                    49930
                } else {
                    49932
                }
            } else if upper.contains("SOUTH POLAR") {
                if sphere {
                    49935
                } else {
                    49937
                }
            } else if upper.starts_with("GCS NAME =") {
                if sphere {
                    49900
                } else {
                    49902
                }
            } else {
                return Err(GisError::InvalidCrs(format!(
                    "unsupported IAU planetary citation {prefix:?}"
                )));
            }
        } else {
            return Ok(None);
        };
        let mut result = HashMap::new();
        result.insert("name".to_string(), "IAU".to_string());
        result.insert("code".to_string(), code.to_string());
        return Ok(Some(result));
    };
    if !authority.eq_ignore_ascii_case("IAU") || !code.chars().all(|ch| ch.is_ascii_digit()) {
        return Ok(None);
    }
    let mut result = HashMap::new();
    result.insert("name".to_string(), "IAU".to_string());
    result.insert("code".to_string(), code.to_string());
    Ok(Some(result))
}

fn extract_ascii_range(ascii: &str, offset: u16, count: u16) -> Option<String> {
    let start = offset as usize;
    let end = start.checked_add(count as usize)?;
    let bytes = ascii.as_bytes();
    if start >= bytes.len() || end > bytes.len() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&bytes[start..end])
            .trim_matches(char::from(0))
            .trim_matches('|')
            .trim()
            .to_string(),
    )
}

fn extract_wkt_from_ascii(ascii: &str) -> Option<String> {
    ascii
        .split('|')
        .map(str::trim)
        .find(|value| looks_like_wkt(value))
        .map(str::to_string)
}

fn validated_wkt_from_ascii(ascii: &str) -> GisResult<Option<String>> {
    extract_wkt_from_ascii(ascii)
        .map(|wkt| validate_wkt_literal(&wkt))
        .transpose()
}

fn validate_wkt_literal(value: &str) -> GisResult<String> {
    let trimmed = value.trim();
    if !looks_like_wkt(trimmed) {
        return Err(GisError::InvalidCrs(
            "CRS WKT must start with a supported WKT CRS token".to_string(),
        ));
    }
    let mut depth = 0i32;
    for ch in trimmed.chars() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth < 0 {
                    return Err(GisError::InvalidCrs(
                        "CRS WKT has unbalanced brackets".to_string(),
                    ));
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(GisError::InvalidCrs(
            "CRS WKT has unbalanced brackets".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

fn looks_like_wkt(value: &str) -> bool {
    let upper = value.trim_start().to_ascii_uppercase();
    ["GEOGCRS[", "PROJCRS[", "GEOGCS[", "PROJCS["]
        .iter()
        .any(|prefix| upper.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::{iau_authority_from_ascii, read_crs, read_transform};
    use std::io::Cursor;
    use tiff::decoder::Decoder;
    use tiff::encoder::{colortype, TiffEncoder};
    use tiff::tags::Tag;

    #[test]
    fn unsupported_iau_citation_is_preserved_for_metadata_only_reads() {
        let authority = iau_authority_from_ascii(
            "IAU:30199|GCS Name = IAU:30199|Ellipsoid = Moon_2000_IAU_IAG|",
        )
        .unwrap()
        .expect("numeric IAU citation");
        assert_eq!(authority.get("name").map(String::as_str), Some("IAU"));
        assert_eq!(authority.get("code").map(String::as_str), Some("30199"));
    }

    #[test]
    fn gdal_iau_2015_citations_resolve_to_curated_codes() {
        let cases = [
            ("GCS Name = Moon (2015) - Sphere / Ocentric|", "30100"),
            (
                "Moon (2015) - Sphere / Ocentric / Equirectangular, clon = 180|",
                "30115",
            ),
            (
                "Mars (2015) / Ocentric / Equirectangular, clon = 0|",
                "49912",
            ),
            ("Mars (2015) - Sphere / Ocentric / North Polar|", "49930"),
            ("Mars (2015) / Ocentric / South Polar|", "49937"),
            ("Mars (2015) / Ographic / North Polar|", "49931"),
        ];
        for (citation, expected) in cases {
            let authority = iau_authority_from_ascii(citation).unwrap().unwrap();
            assert_eq!(authority.get("code").map(String::as_str), Some(expected));
        }
        assert!(iau_authority_from_ascii("Mars (2015) / Ocentric / Orthographic|").is_err());
    }

    #[test]
    fn iau_citation_uses_geokey_ascii_offset() {
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut bytes).unwrap();
            let mut image = encoder.new_image::<colortype::Gray8>(1, 1).unwrap();
            image
                .encoder()
                .write_tag(Tag::GeoAsciiParamsTag, "unrelated|IAU:30115|")
                .unwrap();
            let geokeys = [1_u16, 1, 0, 1, 3073, 34737, 10, 10];
            image
                .encoder()
                .write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
                .unwrap();
            image.write_data(&[0]).unwrap();
        }
        bytes.set_position(0);
        let mut decoder = Decoder::new(bytes).unwrap();
        let (_, authority) = read_crs(&mut decoder).unwrap().unwrap();
        let authority = authority.unwrap();
        assert_eq!(authority.get("name").map(String::as_str), Some("IAU"));
        assert_eq!(authority.get("code").map(String::as_str), Some("30115"));
    }

    #[test]
    fn geoascii_only_iau_preserves_defining_wkt() {
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut bytes).unwrap();
            let mut image = encoder.new_image::<colortype::Gray8>(1, 1).unwrap();
            image
                .encoder()
                .write_tag(
                    Tag::GeoAsciiParamsTag,
                    "PROJCRS[\"Unknown lunar projection\"]|IAU:30199|",
                )
                .unwrap();
            image.write_data(&[0]).unwrap();
        }
        bytes.set_position(0);
        let mut decoder = Decoder::new(bytes).unwrap();
        let (wkt, authority) = read_crs(&mut decoder).unwrap().unwrap();
        assert_eq!(
            wkt.as_deref(),
            Some("PROJCRS[\"Unknown lunar projection\"]")
        );
        assert_eq!(
            authority.unwrap().get("code").map(String::as_str),
            Some("30199")
        );
    }

    #[test]
    fn projected_iau_citation_wins_over_base_geodetic_citation() {
        let projected = "Mars (2015) / Ocentric / Equirectangular, clon = 0|";
        let geographic = "GCS Name = Mars (2015) / Ocentric|";
        let ascii = format!("{projected}{geographic}");
        let projected_count = u16::try_from(projected.len()).unwrap();
        let geographic_count = u16::try_from(geographic.len()).unwrap();
        let geokeys = [
            1_u16,
            1,
            0,
            2,
            1026,
            34737,
            projected_count,
            0,
            2049,
            34737,
            geographic_count,
            projected_count,
        ];
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut bytes).unwrap();
            let mut image = encoder.new_image::<colortype::Gray8>(1, 1).unwrap();
            image
                .encoder()
                .write_tag(Tag::GeoAsciiParamsTag, ascii.as_str())
                .unwrap();
            image
                .encoder()
                .write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
                .unwrap();
            image.write_data(&[0]).unwrap();
        }
        bytes.set_position(0);
        let mut decoder = Decoder::new(bytes).unwrap();
        let (_, authority) = read_crs(&mut decoder).unwrap().unwrap();
        let authority = authority.unwrap();
        assert_eq!(authority.get("code").map(String::as_str), Some("49912"));
    }

    fn decoder_with_tags(
        pixel_scale: Option<&[f64]>,
        tiepoint: Option<&[f64]>,
        matrix: Option<&[f64]>,
        epsg: Option<u16>,
    ) -> Decoder<Cursor<Vec<u8>>> {
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut bytes).unwrap();
            let mut image = encoder.new_image::<colortype::Gray8>(1, 1).unwrap();
            if let Some(values) = pixel_scale {
                image
                    .encoder()
                    .write_tag(Tag::ModelPixelScaleTag, values)
                    .unwrap();
            }
            if let Some(values) = tiepoint {
                image
                    .encoder()
                    .write_tag(Tag::ModelTiepointTag, values)
                    .unwrap();
            }
            if let Some(values) = matrix {
                image
                    .encoder()
                    .write_tag(Tag::ModelTransformationTag, values)
                    .unwrap();
            }
            if let Some(code) = epsg {
                let geokeys = [1_u16, 1, 0, 1, 3072, 0, 1, code];
                image
                    .encoder()
                    .write_tag(Tag::GeoKeyDirectoryTag, &geokeys[..])
                    .unwrap();
            }
            image.write_data(&[0]).unwrap();
        }
        bytes.set_position(0);
        Decoder::new(bytes).unwrap()
    }

    fn valid_matrix() -> [f64; 16] {
        [
            2.0,
            0.0,
            0.0,
            500_000.0,
            0.0,
            -3.0,
            0.0,
            5_500_000.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }

    #[test]
    fn transform_absence_is_the_only_local_fallback() {
        let mut decoder = decoder_with_tags(None, None, None, None);
        assert_eq!(read_transform(&mut decoder).unwrap(), None);
    }

    #[test]
    fn paired_scale_and_tiepoint_produce_north_up_affine() {
        let mut decoder = decoder_with_tags(
            Some(&[2.0, 3.0, 0.0]),
            Some(&[0.0, 0.0, 0.0, 500_000.0, 5_500_000.0, 0.0]),
            None,
            None,
        );
        assert_eq!(
            read_transform(&mut decoder).unwrap().unwrap().tuple(),
            (2.0, 0.0, 500_000.0, 0.0, -3.0, 5_500_000.0)
        );
    }

    #[test]
    fn partial_scale_or_tiepoint_metadata_is_rejected() {
        let mut scale_only = decoder_with_tags(Some(&[1.0, 1.0, 0.0]), None, None, None);
        assert!(read_transform(&mut scale_only).is_err());
        let mut tiepoint_only =
            decoder_with_tags(None, Some(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), None, None);
        assert!(read_transform(&mut tiepoint_only).is_err());
    }

    #[test]
    fn every_rejected_scale_tiepoint_shape_leaves_resource_ledger_exactly_stable() {
        let before = crate::core::resource_tracker::ledger_snapshot();
        let cases: &[(Option<&[f64]>, Option<&[f64]>)] = &[
            (Some(&[1.0, 1.0]), Some(&[0.0; 6])),
            (Some(&[1.0, 1.0, 0.0, 9.0]), Some(&[0.0; 6])),
            (Some(&[1.0, 1.0, 0.0]), Some(&[0.0; 5])),
            (Some(&[1.0, 1.0, 0.0]), Some(&[0.0; 12])),
            (Some(&[f64::NAN, 1.0, 0.0]), Some(&[0.0; 6])),
            (Some(&[-1.0, 1.0, 0.0]), Some(&[0.0; 6])),
            (Some(&[1.0, 0.0, 0.0]), Some(&[0.0; 6])),
        ];
        for (pixel_scale, tiepoint) in cases {
            let mut decoder = decoder_with_tags(*pixel_scale, *tiepoint, None, None);
            assert!(read_transform(&mut decoder).is_err());
            let after = crate::core::resource_tracker::ledger_snapshot();
            assert_eq!(
                after.current_host_visible_bytes,
                before.current_host_visible_bytes
            );
            assert_eq!(
                after.current_device_local_bytes,
                before.current_device_local_bytes
            );
            assert_eq!(after.by_label, before.by_label);
        }
    }

    #[test]
    fn transformation_length_must_be_exactly_sixteen() {
        for len in [15, 17] {
            let values = vec![0.0; len];
            let mut decoder = decoder_with_tags(None, None, Some(&values), None);
            assert!(
                read_transform(&mut decoder).is_err(),
                "accepted length {len}"
            );
        }
    }

    #[test]
    fn malformed_transformations_are_rejected_but_orientation_is_metadata() {
        let mut rejected_cases = Vec::new();
        let mut non_finite = valid_matrix();
        non_finite[0] = f64::NAN;
        rejected_cases.push(non_finite);
        let mut zero_span = valid_matrix();
        zero_span[0] = 0.0;
        rejected_cases.push(zero_span);

        for matrix in rejected_cases {
            let mut decoder = decoder_with_tags(None, None, Some(&matrix), None);
            assert!(read_transform(&mut decoder).is_err(), "accepted {matrix:?}");
        }

        let mut accepted_cases = Vec::new();
        let mut mirrored = valid_matrix();
        mirrored[0] = -2.0;
        accepted_cases.push(mirrored);
        let mut south_up = valid_matrix();
        south_up[5] = 3.0;
        accepted_cases.push(south_up);
        let mut rotated = valid_matrix();
        rotated[1] = 0.25;
        accepted_cases.push(rotated);
        let mut sheared = valid_matrix();
        sheared[4] = 0.25;
        accepted_cases.push(sheared);

        for matrix in accepted_cases {
            let mut decoder = decoder_with_tags(None, None, Some(&matrix), None);
            assert!(read_transform(&mut decoder).is_ok(), "rejected {matrix:?}");
        }
    }

    #[test]
    fn valid_transform_without_crs_is_unlabeled_cartesian() {
        let matrix = valid_matrix();
        let mut decoder = decoder_with_tags(None, None, Some(&matrix), None);
        assert!(read_transform(&mut decoder).unwrap().is_some());
        assert!(read_crs(&mut decoder).unwrap().is_none());
    }

    #[test]
    fn arbitrary_epsg_authority_is_preserved_without_invented_wkt() {
        let matrix = valid_matrix();
        let mut decoder = decoder_with_tags(None, None, Some(&matrix), Some(32633));
        assert!(read_transform(&mut decoder).unwrap().is_some());
        let (wkt, authority) = read_crs(&mut decoder).unwrap().unwrap();
        assert!(wkt.is_none());
        let authority = authority.unwrap();
        assert_eq!(authority.get("name").map(String::as_str), Some("EPSG"));
        assert_eq!(authority.get("code").map(String::as_str), Some("32633"));
    }
}
