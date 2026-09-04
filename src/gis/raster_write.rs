use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use tiff::encoder::colortype;
use tiff::encoder::compression::{Compression, Deflate, Lzw, Packbits, Uncompressed};
use tiff::encoder::{DirectoryEncoder, TiffEncoder, TiffKind, TiffValue};
use tiff::tags::Tag;

use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info::read_raster_info;
use crate::gis::types::{AffineTransform, RasterDType, RasterInfo};

const FORGE3D_NODATA_PREFIX: &str = "forge3d:nodata_per_band=";
/// Private TIFF ASCII tag (32768–65535 range) carrying the MENSURA vertical
/// datum, so `height_system` survives a GeoTIFF write→read round trip.
pub(crate) const FORGE3D_HEIGHT_SYSTEM_TAG: u16 = 65001;

#[derive(Debug, Clone)]
pub enum RasterData {
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl RasterData {
    pub fn dtype(&self) -> RasterDType {
        match self {
            RasterData::U8(_) => RasterDType::UInt8,
            RasterData::I16(_) => RasterDType::Int16,
            RasterData::U16(_) => RasterDType::UInt16,
            RasterData::I32(_) => RasterDType::Int32,
            RasterData::U32(_) => RasterDType::UInt32,
            RasterData::F32(_) => RasterDType::Float32,
            RasterData::F64(_) => RasterDType::Float64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RasterArray {
    pub data: RasterData,
    pub bands: usize,
    pub height: usize,
    pub width: usize,
}

impl RasterArray {
    pub fn new(data: RasterData, shape: &[usize]) -> GisResult<Self> {
        let (bands, height, width) = validate_shape(shape)?;
        let expected_len = bands
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| GisError::InvalidShape("array shape is too large".to_string()))?;
        let actual_len = match &data {
            RasterData::U8(values) => values.len(),
            RasterData::I16(values) => values.len(),
            RasterData::U16(values) => values.len(),
            RasterData::I32(values) => values.len(),
            RasterData::U32(values) => values.len(),
            RasterData::F32(values) => values.len(),
            RasterData::F64(values) => values.len(),
        };
        if actual_len != expected_len {
            return Err(GisError::InvalidShape(format!(
                "array length {actual_len} does not match shape {:?}",
                shape
            )));
        }
        Ok(Self {
            data,
            bands,
            height,
            width,
        })
    }

    pub fn dtype(&self) -> RasterDType {
        self.data.dtype()
    }
}

#[derive(Debug, Clone)]
pub struct WriteRasterOptions {
    pub crs: Option<CrsSpec>,
    pub transform: Option<AffineTransform>,
    pub nodata: Vec<Option<f64>>,
    pub driver: String,
    pub overwrite: bool,
    pub creation_options: CreationOptions,
    pub creation_options_explicit: bool,
    pub like_info: Option<RasterInfo>,
    /// MENSURA M-03: vertical datum persisted as a private GeoTIFF ASCII tag
    /// (see `FORGE3D_HEIGHT_SYSTEM_TAG`). "unspecified" is not written.
    pub height_system: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrsSpec {
    pub authority: Option<(String, String)>,
    pub wkt: Option<String>,
    pub projection: Option<crate::geo::projections::ProjectionDefinition>,
}

impl CrsSpec {
    pub fn from_string(value: String) -> GisResult<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(GisError::InvalidCrs(
                "CRS string cannot be empty".to_string(),
            ));
        }
        if let Some((authority, code)) = parse_authority_code(trimmed) {
            let (authority, code) = validate_authority_code(authority, code)?;
            return Ok(Self {
                authority: Some((authority, code)),
                wkt: None,
                projection: None,
            });
        }
        if looks_like_wkt(trimmed) {
            return Ok(Self {
                authority: None,
                wkt: Some(validate_wkt_literal(trimmed)?),
                projection: None,
            });
        }
        Err(GisError::InvalidCrs(
            "CRS must be an authority string such as EPSG:4326, a WKT string, or a CRS dict"
                .to_string(),
        ))
    }

    pub fn from_parts(
        authority: Option<String>,
        code: Option<String>,
        wkt: Option<String>,
    ) -> GisResult<Self> {
        let iau_with_wkt = authority
            .as_deref()
            .is_some_and(|name| name.trim().eq_ignore_ascii_case("IAU"))
            && wkt.is_some();
        let authority_pair = match (authority, code) {
            (Some(_), Some(code)) if iau_with_wkt => {
                let code = code.trim().to_string();
                let numeric = code.parse::<u32>().map_err(|_| {
                    GisError::InvalidCrs("CRS authority code must be numeric".to_string())
                })?;
                if crate::geo::projections::iau_is_mars_ographic(numeric) {
                    return Err(GisError::InvalidCrs(format!(
                        "unsupported IAU:{code} Mars planetographic/ographic west-positive CRS; \
                         SELENE supports planetocentric +East only"
                    )));
                }
                Some(("IAU".to_string(), code))
            }
            (Some(authority), Some(code)) => Some(validate_authority_code(&authority, &code)?),
            (None, None) => None,
            _ => {
                return Err(GisError::InvalidCrs(
                    "CRS dict must include both name and code".to_string(),
                ))
            }
        };
        let wkt = wkt
            .map(|value| {
                let trimmed = value.trim();
                if iau_with_wkt {
                    validate_wkt_structure(trimmed)
                } else if looks_like_wkt(trimmed) {
                    validate_wkt_literal(trimmed)
                } else {
                    Err(GisError::InvalidCrs(
                        "CRS WKT dict entry does not look like WKT".to_string(),
                    ))
                }
            })
            .transpose()?;
        if iau_with_wkt {
            let (Some(authority_pair), Some(wkt)) = (authority_pair.as_ref(), wkt.as_deref())
            else {
                return Err(GisError::InvalidCrs(
                    "IAU CRS dict must include both name/code and WKT".to_string(),
                ));
            };
            validate_iau_wkt_body(authority_pair, wkt)?;
        }
        if authority_pair.is_none() && wkt.is_none() {
            return Err(GisError::InvalidCrs(
                "CRS dict must include name/code or WKT".to_string(),
            ));
        }
        if authority_pair.is_some() && wkt.is_some() && !iau_with_wkt {
            return Err(GisError::InvalidCrs(
                "CRS dict must use either name/code or WKT, not both".to_string(),
            ));
        }
        Ok(Self {
            authority: authority_pair,
            wkt,
            projection: None,
        })
    }

    pub fn from_projection(projection: crate::geo::projections::ProjectionDefinition) -> Self {
        Self {
            authority: None,
            wkt: None,
            projection: Some(projection),
        }
    }

    pub(crate) fn from_raster_info(info: &RasterInfo) -> Option<Self> {
        let authority = info.crs_authority.as_ref().and_then(|authority| {
            Some((
                authority.get("name")?.to_ascii_uppercase(),
                authority.get("code")?.to_string(),
            ))
        });
        if authority.is_none() && info.crs_wkt.is_none() {
            None
        } else {
            Some(Self {
                authority,
                wkt: info.crs_wkt.clone(),
                projection: None,
            })
        }
    }

    fn equivalent_to_info(&self, info: &RasterInfo) -> bool {
        let info_spec = Self::from_raster_info(info);
        match info_spec {
            Some(info_spec) => self.equivalent_to(&info_spec),
            None => false,
        }
    }

    pub(crate) fn equivalent_to(&self, other: &Self) -> bool {
        self == other
            || self.same_supported_iau_authority(other)
            || (self.has_epsg_4326_authority() && other.has_wgs84_wkt())
            || (other.has_epsg_4326_authority() && self.has_wgs84_wkt())
    }

    fn same_supported_iau_authority(&self, other: &Self) -> bool {
        let (Some((left_name, left_code)), Some((right_name, right_code))) =
            (&self.authority, &other.authority)
        else {
            return false;
        };
        left_name.eq_ignore_ascii_case("IAU")
            && right_name.eq_ignore_ascii_case("IAU")
            && left_code == right_code
            && left_code
                .parse()
                .ok()
                .is_some_and(|code| crate::geo::projections::iau_body(code).is_some())
    }

    fn has_epsg_4326_authority(&self) -> bool {
        self.authority.as_ref().is_some_and(|(authority, code)| {
            authority.eq_ignore_ascii_case("EPSG") && code == "4326"
        })
    }

    fn has_wgs84_wkt(&self) -> bool {
        self.wkt
            .as_deref()
            .is_some_and(|wkt| validate_wkt_literal(wkt).is_ok())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionChoice {
    None,
    Lzw,
    Deflate,
    Packbits,
}

#[derive(Debug, Clone)]
pub struct CreationOptions {
    pub compression: CompressionChoice,
    pub bigtiff: bool,
}

impl Default for CreationOptions {
    fn default() -> Self {
        Self {
            compression: CompressionChoice::None,
            bigtiff: false,
        }
    }
}

impl CreationOptions {
    pub fn from_map(values: &HashMap<String, String>) -> GisResult<Self> {
        let mut options = Self::default();
        for (key, value) in values {
            match key.as_str() {
                "compress" => {
                    options.compression = match value.to_ascii_uppercase().as_str() {
                        "" | "NONE" | "UNCOMPRESSED" => CompressionChoice::None,
                        "LZW" => CompressionChoice::Lzw,
                        "DEFLATE" | "ZLIB" => CompressionChoice::Deflate,
                        "PACKBITS" => CompressionChoice::Packbits,
                        other => {
                            return Err(GisError::InvalidArgument(format!(
                                "unsupported compress value {other:?}"
                            )))
                        }
                    };
                }
                "tiled" => {
                    if parse_bool(value)? {
                        return Err(GisError::InvalidArgument(
                            "tiled GeoTIFF output is not supported in G-002a1".to_string(),
                        ));
                    }
                }
                "blockxsize" | "blockysize" | "predictor" => {
                    return Err(GisError::InvalidArgument(format!(
                        "creation option {key:?} is not supported in G-002a1"
                    )));
                }
                "bigtiff" => {
                    options.bigtiff = parse_bool(value)?;
                    if options.bigtiff {
                        return Err(GisError::InvalidArgument(
                            "BigTIFF output is not supported in G-002a1".to_string(),
                        ));
                    }
                }
                unknown => {
                    return Err(GisError::UnsupportedCreationOption(format!(
                        "unsupported creation option {unknown:?}"
                    )));
                }
            }
        }
        Ok(options)
    }
}

pub fn write_raster(
    path: impl AsRef<Path>,
    array: RasterArray,
    mut options: WriteRasterOptions,
) -> GisResult<RasterInfo> {
    let path = path.as_ref();
    validate_driver(&options.driver)?;
    validate_output_path(path, options.overwrite)?;
    apply_like_info(&array, &mut options)?;
    validate_nodata(&array, &options.nodata)?;

    let target_path = path.to_path_buf();
    let write_path = if options.overwrite {
        temporary_sibling_path(path)
    } else {
        target_path.clone()
    };

    if let Err(err) = write_tiff(&write_path, &array, &options) {
        let _ = fs::remove_file(&write_path);
        return Err(err);
    }

    let mut info = match read_raster_info(&write_path)
        .map_err(|err| GisError::PostWriteValidationFailed(err.to_string()))
        .and_then(|info| {
            validate_post_write(&info, &array, &options, &write_path)?;
            Ok(info)
        }) {
        Ok(info) => info,
        Err(err) => {
            let _ = fs::remove_file(&write_path);
            return Err(err);
        }
    };

    if options.overwrite {
        replace_file(&write_path, &target_path).inspect_err(|_| {
            let _ = fs::remove_file(&write_path);
        })?;
        info = read_raster_info(&target_path)
            .map_err(|err| GisError::PostWriteValidationFailed(err.to_string()))?;
        validate_post_write(&info, &array, &options, &target_path)?;
    }

    Ok(info)
}

fn validate_shape(shape: &[usize]) -> GisResult<(usize, usize, usize)> {
    match shape {
        [height, width] if *height > 0 && *width > 0 => Ok((1, *height, *width)),
        [first, second, third] if *first > 0 && *second > 0 && *third > 0 => {
            Ok((*first, *second, *third))
        }
        _ => Err(GisError::InvalidShape(
            "expected shape (height, width) or (bands, height, width)".to_string(),
        )),
    }
}

fn validate_driver(driver: &str) -> GisResult<()> {
    if driver.eq_ignore_ascii_case("GTiff") || driver.eq_ignore_ascii_case("GeoTIFF") {
        Ok(())
    } else {
        Err(GisError::UnsupportedDriver(format!(
            "only GTiff is supported, got {driver:?}"
        )))
    }
}

fn validate_output_path(path: &Path, overwrite: bool) -> GisResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(GisError::NotFound(parent.to_path_buf()));
        }
    }
    if path.exists() && !overwrite {
        return Err(GisError::AlreadyExists(path.to_path_buf()));
    }
    Ok(())
}

fn apply_like_info(array: &RasterArray, options: &mut WriteRasterOptions) -> GisResult<()> {
    let Some(like) = options.like_info.clone() else {
        return Ok(());
    };

    if like.transform.is_some()
        && (like.width as usize != array.width || like.height as usize != array.height)
    {
        return Err(GisError::ShapeMismatch(format!(
            "like raster shape is {}x{}, output shape is {}x{}",
            like.width, like.height, array.width, array.height
        )));
    }

    if options.transform.is_none() && like.transform.is_some() {
        if let Some(transform) = like.transform {
            options.transform = Some(AffineTransform::new([
                transform.0,
                transform.1,
                transform.2,
                transform.3,
                transform.4,
                transform.5,
            ])?);
        }
    } else if let (Some(explicit), Some(like_transform)) = (options.transform, like.transform) {
        let like_transform = AffineTransform::new([
            like_transform.0,
            like_transform.1,
            like_transform.2,
            like_transform.3,
            like_transform.4,
            like_transform.5,
        ])?;
        if !affine_eq(explicit, like_transform) {
            return Err(GisError::InvalidArgument(
                "explicit transform conflicts with like_* transform".to_string(),
            ));
        }
    }

    let like_crs = CrsSpec::from_raster_info(&like);
    if options.crs.is_none() {
        options.crs = like_crs;
    } else if like_crs.is_some()
        && !options
            .crs
            .as_ref()
            .is_some_and(|crs| crs.equivalent_to_info(&like))
    {
        return Err(GisError::InvalidArgument(
            "explicit CRS conflicts with like_* CRS".to_string(),
        ));
    }

    if options.nodata.iter().all(Option::is_none)
        && like.nodata_per_band.iter().any(Option::is_some)
    {
        if like.nodata_per_band.len() == array.bands {
            options.nodata = like.nodata_per_band.clone();
        } else if let Some(value) = like.nodata_per_band.first().copied().flatten() {
            options.nodata = vec![Some(value); array.bands];
        }
    }

    if !options.creation_options_explicit {
        if let Some(compression) = like.compression.as_deref() {
            options.creation_options.compression = compression_choice_from_name(compression)?;
        }
    }

    Ok(())
}

fn affine_eq(left: AffineTransform, right: AffineTransform) -> bool {
    let left = left.tuple();
    let right = right.tuple();
    [left.0, left.1, left.2, left.3, left.4, left.5]
        .iter()
        .zip([right.0, right.1, right.2, right.3, right.4, right.5])
        .all(|(a, b)| (a - b).abs() <= 1e-12)
}

fn validate_nodata(array: &RasterArray, nodata: &[Option<f64>]) -> GisResult<()> {
    if nodata.len() != array.bands {
        return Err(GisError::InvalidNodata(format!(
            "nodata length {} does not match band count {}",
            nodata.len(),
            array.bands
        )));
    }
    let dtype = array.dtype();
    for value in nodata.iter().flatten() {
        if !dtype.nodata_fits(*value) {
            return Err(GisError::InvalidNodata(format!(
                "nodata value {value:?} is incompatible with dtype {}",
                dtype.name()
            )));
        }
    }
    Ok(())
}

fn write_tiff(path: &Path, array: &RasterArray, options: &WriteRasterOptions) -> GisResult<()> {
    let file = File::create(path).map_err(|err| GisError::WriteFailed(err.to_string()))?;
    let mut encoder =
        TiffEncoder::new(file).map_err(|err| GisError::WriteFailed(err.to_string()))?;
    match options.creation_options.compression {
        CompressionChoice::None => {
            write_tiff_with_compression::<Uncompressed>(&mut encoder, array, options)
        }
        CompressionChoice::Lzw => write_tiff_with_compression::<Lzw>(&mut encoder, array, options),
        CompressionChoice::Deflate => {
            write_tiff_with_compression::<Deflate>(&mut encoder, array, options)
        }
        CompressionChoice::Packbits => {
            write_tiff_with_compression::<Packbits>(&mut encoder, array, options)
        }
    }
}

fn write_tiff_with_compression<D>(
    encoder: &mut TiffEncoder<File>,
    array: &RasterArray,
    options: &WriteRasterOptions,
) -> GisResult<()>
where
    D: Compression + Default,
{
    match &array.data {
        RasterData::U8(data) => match array.bands {
            1 => write_image::<colortype::Gray8, u8, D>(encoder, array, options, data),
            3 => write_image::<colortype::RGB8, u8, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            4 => write_image::<colortype::RGBA8, u8, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            _ => write_manual_if_uncompressed::<u8>(encoder, array, options, data),
        },
        RasterData::I16(data) => match array.bands {
            1 => write_image::<colortype::GrayI16, i16, D>(encoder, array, options, data),
            _ => write_manual_if_uncompressed::<i16>(encoder, array, options, data),
        },
        RasterData::U16(data) => match array.bands {
            1 => write_image::<colortype::Gray16, u16, D>(encoder, array, options, data),
            3 => write_image::<colortype::RGB16, u16, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            4 => write_image::<colortype::RGBA16, u16, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            _ => write_manual_if_uncompressed::<u16>(encoder, array, options, data),
        },
        RasterData::I32(data) => match array.bands {
            1 => write_image::<colortype::GrayI32, i32, D>(encoder, array, options, data),
            _ => write_manual_if_uncompressed::<i32>(encoder, array, options, data),
        },
        RasterData::U32(data) => match array.bands {
            1 => write_image::<colortype::Gray32, u32, D>(encoder, array, options, data),
            3 => write_image::<colortype::RGB32, u32, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            4 => write_image::<colortype::RGBA32, u32, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            _ => write_manual_if_uncompressed::<u32>(encoder, array, options, data),
        },
        RasterData::F32(data) => match array.bands {
            1 => write_image::<colortype::Gray32Float, f32, D>(encoder, array, options, data),
            3 => write_image::<colortype::RGB32Float, f32, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            4 => write_image::<colortype::RGBA32Float, f32, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            _ => write_manual_if_uncompressed::<f32>(encoder, array, options, data),
        },
        RasterData::F64(data) => match array.bands {
            1 => write_image::<colortype::Gray64Float, f64, D>(encoder, array, options, data),
            3 => write_image::<colortype::RGB64Float, f64, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            4 => write_image::<colortype::RGBA64Float, f64, D>(
                encoder,
                array,
                options,
                &interleave_bhw(data, array),
            ),
            _ => write_manual_if_uncompressed::<f64>(encoder, array, options, data),
        },
    }
}

fn write_image<C, T, D>(
    encoder: &mut TiffEncoder<File>,
    array: &RasterArray,
    options: &WriteRasterOptions,
    data: &[T],
) -> GisResult<()>
where
    C: colortype::ColorType<Inner = T>,
    T: TiffValue + Copy,
    [T]: TiffValue,
    D: Compression + Default,
{
    let mut image = encoder
        .new_image_with_compression::<C, D>(array.width as u32, array.height as u32, D::default())
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    write_geotiff_tags(image.encoder(), options)?;
    image
        .write_data(data)
        .map_err(|err| GisError::WriteFailed(err.to_string()))
}

fn write_manual_if_uncompressed<T>(
    encoder: &mut TiffEncoder<File>,
    array: &RasterArray,
    options: &WriteRasterOptions,
    data: &[T],
) -> GisResult<()>
where
    T: TiffValue + Copy,
    [T]: TiffValue,
{
    if options.creation_options.compression != CompressionChoice::None {
        return Err(GisError::InvalidArgument(
            "non-standard band counts only support uncompressed output in G-002a1".to_string(),
        ));
    }
    let interleaved = interleave_bhw(data, array);
    let mut directory = encoder
        .new_directory()
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    let strip_offset = directory
        .write_data(interleaved.as_slice())
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    let strip_offset = u32::try_from(strip_offset).map_err(|_| {
        GisError::WriteFailed("TIFF strip offset exceeds classic TIFF range".to_string())
    })?;
    let byte_count = interleaved
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            GisError::WriteFailed("TIFF strip byte count exceeds classic TIFF range".to_string())
        })?;
    let band_count = u16::try_from(array.bands)
        .map_err(|_| GisError::InvalidShape("band count exceeds u16 range".to_string()))?;
    let bits = vec![dtype_bits(array.dtype()); array.bands];
    let sample_formats = vec![dtype_sample_format(array.dtype()); array.bands];

    directory
        .write_tag(Tag::ImageWidth, array.width as u32)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::ImageLength, array.height as u32)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::BitsPerSample, bits.as_slice())
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::Compression, 1u16)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::PhotometricInterpretation, 1u16)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::StripOffsets, strip_offset)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::SamplesPerPixel, band_count)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::RowsPerStrip, array.height as u32)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::StripByteCounts, byte_count)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::PlanarConfiguration, 1u16)
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    directory
        .write_tag(Tag::SampleFormat, sample_formats.as_slice())
        .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    write_geotiff_tags(&mut directory, options)?;
    directory
        .finish()
        .map_err(|err| GisError::WriteFailed(err.to_string()))
}

fn write_geotiff_tags<W, K>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    options: &WriteRasterOptions,
) -> GisResult<()>
where
    W: std::io::Write + std::io::Seek,
    K: TiffKind,
{
    if let Some(transform) = options.transform {
        if transform_requires_matrix(transform) {
            let matrix = [
                transform.a,
                transform.b,
                0.0,
                transform.c,
                transform.d,
                transform.e,
                0.0,
                transform.f,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ];
            directory
                .write_tag(Tag::ModelTransformationTag, &matrix[..])
                .map_err(|err| GisError::WriteFailed(err.to_string()))?;
        } else {
            let pixel_scale = [transform.a.abs(), transform.e.abs(), 0.0];
            let tiepoint = [0.0, 0.0, 0.0, transform.c, transform.f, 0.0];
            directory
                .write_tag(Tag::ModelPixelScaleTag, &pixel_scale[..])
                .map_err(|err| GisError::WriteFailed(err.to_string()))?;
            directory
                .write_tag(Tag::ModelTiepointTag, &tiepoint[..])
                .map_err(|err| GisError::WriteFailed(err.to_string()))?;
        }
    }

    if let Some(crs) = &options.crs {
        let crs_ascii = crs_ascii_for_crs(crs);
        if let Some(crs_ascii) = &crs_ascii {
            directory
                .write_tag(Tag::GeoAsciiParamsTag, crs_ascii.as_str())
                .map_err(|err| GisError::WriteFailed(err.to_string()))?;
        }
        let tags = geokeys_for_crs(crs)?;
        if let Some(tags) = tags {
            if !tags.doubles.is_empty() {
                directory
                    .write_tag(Tag::GeoDoubleParamsTag, &tags.doubles[..])
                    .map_err(|err| GisError::WriteFailed(err.to_string()))?;
            }
            directory
                .write_tag(Tag::GeoKeyDirectoryTag, &tags.directory[..])
                .map_err(|err| GisError::WriteFailed(err.to_string()))?;
        }
    }

    if let Some(nodata) = uniform_nodata(&options.nodata) {
        let value = if nodata.is_nan() {
            "nan".to_string()
        } else {
            format!("{nodata}")
        };
        directory
            .write_tag(Tag::GdalNodata, value.as_str())
            .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    } else if options.nodata.iter().any(Option::is_some) {
        let encoded = format!(
            "{FORGE3D_NODATA_PREFIX}{}",
            options
                .nodata
                .iter()
                .map(|value| match value {
                    Some(value) if value.is_nan() => "nan".to_string(),
                    Some(value) => format!("{value}"),
                    None => "none".to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
        );
        directory
            .write_tag(Tag::ImageDescription, encoded.as_str())
            .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    }

    // MENSURA M-03: persist a declared vertical datum. "unspecified" is the
    // absent state and is never written (an absent tag reads back as such).
    if crate::gis::types::is_valid_height_system(&options.height_system)
        && options.height_system != crate::gis::types::HEIGHT_SYSTEM_UNSPECIFIED
    {
        directory
            .write_tag(
                Tag::Unknown(FORGE3D_HEIGHT_SYSTEM_TAG),
                options.height_system.as_str(),
            )
            .map_err(|err| GisError::WriteFailed(err.to_string()))?;
    }

    Ok(())
}

fn transform_requires_matrix(transform: AffineTransform) -> bool {
    transform.is_rotated_or_sheared() || transform.a < 0.0 || transform.e > 0.0
}

struct GeoKeyTags {
    directory: Vec<u16>,
    doubles: Vec<f64>,
}

fn crs_ascii_for_crs(crs: &CrsSpec) -> Option<String> {
    let iau = crs
        .authority
        .as_ref()
        .filter(|(authority, _)| authority.eq_ignore_ascii_case("IAU"))
        .map(|(_, code)| format!("IAU:{code}|"));
    match (iau, crs.wkt.as_ref()) {
        (Some(iau), Some(wkt)) => Some(format!("{iau}{wkt}|")),
        (Some(iau), None) => Some(iau),
        (None, Some(wkt)) => Some(format!("{wkt}|")),
        (None, None) => None,
    }
}

fn geokeys_for_crs(crs: &CrsSpec) -> GisResult<Option<GeoKeyTags>> {
    let iau_count = crs
        .authority
        .as_ref()
        .filter(|(authority, _)| authority.eq_ignore_ascii_case("IAU"))
        .map(|(_, code)| format!("IAU:{code}|").len())
        .map(|value| {
            u16::try_from(value).map_err(|_| {
                GisError::InvalidCrs("IAU citation is too large for GeoTIFF ASCII key".to_string())
            })
        })
        .transpose()?;
    let wkt_count = crs
        .wkt
        .as_ref()
        .map(|wkt| wkt.len() + 1)
        .map(|value| {
            u16::try_from(value).map_err(|_| {
                GisError::InvalidCrs("CRS WKT is too large for GeoTIFF ASCII key".to_string())
            })
        })
        .transpose()?;
    let wkt_offset = iau_count.unwrap_or(0);

    if let Some((authority, code)) = &crs.authority {
        let normalized_authority = authority.trim().to_ascii_uppercase();
        if normalized_authority == "IAU" {
            let numeric: u32 = code
                .parse()
                .map_err(|_| GisError::InvalidCrs(format!("invalid IAU code {code:?}")))?;
            let Some(body) = crate::geo::projections::iau_body(numeric) else {
                let wkt = crs.wkt.as_deref().ok_or_else(|| {
                    GisError::InvalidCrs(format!(
                        "unsupported IAU code {code} cannot be written without a defining WKT"
                    ))
                })?;
                let upper = wkt.trim_start().to_ascii_uppercase();
                let model_type = if upper.starts_with("PROJCRS[") || upper.starts_with("PROJCS[") {
                    1
                } else if upper.starts_with("GEOGCRS[") || upper.starts_with("GEOGCS[") {
                    2
                } else {
                    return Err(GisError::InvalidCrs(format!(
                        "unsupported IAU code {code} has an unclassifiable WKT"
                    )));
                };
                let mut entries = vec![1024, 0, 1, model_type, 1025, 0, 1, 1];
                if let Some(count) = wkt_count {
                    entries.extend([1026, Tag::GeoAsciiParamsTag.to_u16(), count, wkt_offset]);
                }
                let citation_key = if model_type == 1 { 3073 } else { 2049 };
                let crs_key = if model_type == 1 { 3072 } else { 2048 };
                entries.extend([crs_key, 0, 1, 32767]);
                entries.extend([
                    citation_key,
                    Tag::GeoAsciiParamsTag.to_u16(),
                    iau_count.unwrap_or(0),
                    0,
                ]);
                return Ok(Some(GeoKeyTags {
                    directory: geokey_directory(entries),
                    doubles: Vec::new(),
                }));
            };
            let ellipsoid = crate::geo::projections::iau_ellipsoid(body, numeric)
                .ok_or_else(|| GisError::InvalidCrs(format!("unsupported IAU code {code}")))?;
            let geographic = matches!(numeric, 30100 | 49900 | 49902);
            let double_tag = Tag::GeoDoubleParamsTag.to_u16();
            let mut doubles = vec![ellipsoid.a, ellipsoid.b()];
            let mut entries = vec![1024, 0, 1, if geographic { 2 } else { 1 }, 1025, 0, 1, 1];
            if let Some(count) = wkt_count {
                entries.extend([1026, Tag::GeoAsciiParamsTag.to_u16(), count, wkt_offset]);
            }
            entries.extend([
                2048,
                0,
                1,
                32767,
                2049,
                Tag::GeoAsciiParamsTag.to_u16(),
                iau_count.unwrap_or(0),
                0,
                2050,
                0,
                1,
                32767,
                2051,
                0,
                1,
                8901,
                2054,
                0,
                1,
                9102,
                2056,
                0,
                1,
                32767,
                2057,
                double_tag,
                1,
                0,
                2058,
                double_tag,
                1,
                1,
            ]);
            if !geographic {
                entries.extend([3072, 0, 1, 32767]);
                entries.extend([
                    3073,
                    Tag::GeoAsciiParamsTag.to_u16(),
                    iau_count.unwrap_or(0),
                    0,
                ]);
                entries.extend([
                    3074,
                    0,
                    1,
                    32767,
                    3075,
                    0,
                    1,
                    if matches!(numeric, 30110 | 30115 | 49910 | 49912 | 49915 | 49917) {
                        17
                    } else {
                        15
                    },
                    3076,
                    0,
                    1,
                    9001,
                ]);
                let lon0 = if matches!(numeric, 30115 | 49915 | 49917) {
                    180.0
                } else {
                    0.0
                };
                let projection_parameters =
                    if matches!(numeric, 30110 | 30115 | 49910 | 49912 | 49915 | 49917) {
                        vec![
                            (3078, 0.0),
                            (3082, 0.0),
                            (3083, 0.0),
                            (3088, lon0),
                            (3089, 0.0),
                        ]
                    } else {
                        let lat0 = if matches!(numeric, 30130 | 49930 | 49932) {
                            90.0
                        } else {
                            -90.0
                        };
                        vec![
                            (3081, lat0),
                            (3082, 0.0),
                            (3083, 0.0),
                            (3092, 1.0),
                            (3095, lon0),
                        ]
                    };
                for (key, value) in projection_parameters {
                    let offset = u16::try_from(doubles.len()).map_err(|_| {
                        GisError::InvalidCrs(
                            "too many GeoTIFF double parameters for planetary CRS".to_string(),
                        )
                    })?;
                    doubles.push(value);
                    entries.extend([key, double_tag, 1, offset]);
                }
            }
            return Ok(Some(GeoKeyTags {
                directory: geokey_directory(entries),
                doubles,
            }));
        }
        let (authority, code) = validate_authority_code(authority, code)?;
        if authority == "FORGE3D" {
            return Err(GisError::InvalidCrs(
                "FORGE3D body-fixed 3D frames cannot label a 2D raster".to_string(),
            ));
        }
        let code: u16 = code
            .parse()
            .map_err(|_| GisError::InvalidCrs(format!("invalid EPSG code {code:?}")))?;
        let mut entries = if code == 4326 {
            vec![
                1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2054, 0, 1, 9102,
            ]
        } else {
            vec![
                1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, code, 3076, 0, 1, 9001,
            ]
        };
        if let Some(count) = wkt_count {
            entries.extend([1026, Tag::GeoAsciiParamsTag.to_u16(), count, 0]);
        }
        return Ok(Some(GeoKeyTags {
            directory: geokey_directory(entries),
            doubles: Vec::new(),
        }));
    }

    let Some(count) = wkt_count else {
        return Ok(None);
    };
    let model_type = if crs
        .wkt
        .as_deref()
        .is_some_and(|wkt| wkt.to_ascii_uppercase().contains("PROJ"))
    {
        1
    } else {
        2
    };
    Ok(Some(GeoKeyTags {
        directory: geokey_directory(vec![
            1024,
            0,
            1,
            model_type,
            1025,
            0,
            1,
            1,
            1026,
            Tag::GeoAsciiParamsTag.to_u16(),
            count,
            0,
        ]),
        doubles: Vec::new(),
    }))
}

fn geokey_directory(entries: Vec<u16>) -> Vec<u16> {
    let count = u16::try_from(entries.len() / 4).unwrap_or(u16::MAX);
    let mut directory = vec![1, 1, 0, count];
    directory.extend(entries);
    directory
}

fn parse_authority_code(crs: &str) -> Option<(&str, &str)> {
    let trimmed = crs.trim();
    let (authority, code) = trimmed.split_once(':')?;
    if authority.is_empty() || code.is_empty() || !code.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((authority, code))
}

fn validate_authority_code(authority: &str, code: &str) -> GisResult<(String, String)> {
    let authority = authority.trim().to_ascii_uppercase();
    let code = code.trim().to_string();
    if code.is_empty() || !code.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(GisError::InvalidCrs(
            "CRS authority code must be numeric".to_string(),
        ));
    }
    if authority == "IAU" {
        let numeric: u32 = code
            .parse()
            .map_err(|_| GisError::InvalidCrs(format!("IAU code {code} is not a valid number")))?;
        if crate::geo::projections::iau_is_mars_ographic(numeric) {
            return Err(GisError::InvalidCrs(format!(
                "unsupported IAU:{code} Mars planetographic/ographic west-positive CRS; \
                 SELENE supports planetocentric +East only"
            )));
        }
        if crate::geo::projections::iau_body(numeric).is_some() {
            return Ok((authority, code));
        }
        return Err(GisError::InvalidCrs(format!("unsupported IAU code {code}")));
    }
    if authority == "FORGE3D" {
        return match code.as_str() {
            "301" | "499" => Ok((authority, code)),
            _ => Err(GisError::InvalidCrs(format!(
                "unsupported FORGE3D body-fixed code {code}"
            ))),
        };
    }
    if authority != "EPSG" {
        return Err(GisError::InvalidCrs(
            "CRS authority must be EPSG, IAU, or FORGE3D with a numeric code".to_string(),
        ));
    }
    // MENSURA: projected-CRS support is defined by the one authoritative
    // projection table (`crate::geo::projections::epsg_projection_definition`);
    // this check delegates to it so the two never drift. Geographic-family
    // codes (4000-4999, incl. 4326 and 4979) parse so downstream code can reject
    // them with a precise diagnostic (e.g. geometry_measure's degree guard).
    let numeric: u32 = code
        .parse()
        .map_err(|_| GisError::InvalidCrs(format!("EPSG code {code} is not a valid number")))?;
    let supported = (4000..5000).contains(&numeric)
        || crate::geo::projections::epsg_projection_definition(numeric).is_some();
    if supported {
        Ok((authority, code))
    } else {
        Err(GisError::InvalidCrs(format!(
            "unsupported EPSG code {code}; the built-in projection engine supports EPSG:4326, \
             the WGS84 UTM zones (EPSG:326zz/327zz), and a curated set of projected CRSs \
             (3857, 3395, 2154, 5070, 5041, 5042)"
        )))
    }
}

fn validate_wkt_literal(value: &str) -> GisResult<String> {
    let trimmed = validate_wkt_structure(value)?;
    let upper = trimmed.to_ascii_uppercase();
    // G-002a1 has no general Earth CRS parser; validate only the WGS84-style WKT subset used here.
    if !(upper.contains("WGS 84") || upper.contains("WGS_1984"))
        || !(upper.contains("DATUM[") || upper.contains("ENSEMBLE["))
    {
        return Err(GisError::InvalidCrs(
            "G-002a1 only validates WGS 84 WKT; use EPSG:4326 or EPSG:3857 for authority CRS"
                .to_string(),
        ));
    }
    Ok(trimmed)
}

fn validate_wkt_structure(value: &str) -> GisResult<String> {
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

fn validate_iau_wkt_body(authority: &(String, String), wkt: &str) -> GisResult<()> {
    let code = authority
        .1
        .parse::<u32>()
        .map_err(|_| GisError::InvalidCrs("IAU authority code must be numeric".to_string()))?;
    let expected = crate::geo::projections::iau_body(code).or(match code / 100 {
        301 => Some(&crate::geo::body::MOON),
        499 => Some(&crate::geo::body::MARS),
        _ => None,
    });
    let Some(expected) = expected else {
        return Ok(());
    };
    let upper = wkt.to_ascii_uppercase();
    let names_moon = upper.contains("MOON") || upper.contains("LUNAR");
    let names_mars = upper.contains("MARS") || upper.contains("MARTIAN");
    let names_earth =
        upper.contains("EARTH") || upper.contains("WGS 84") || upper.contains("WGS_1984");
    let names_expected = if expected == &crate::geo::body::MOON {
        names_moon
    } else {
        names_mars
    };
    let names_other = if expected == &crate::geo::body::MOON {
        names_mars
    } else {
        names_moon
    };
    if !names_expected || names_other || names_earth {
        return Err(GisError::InvalidCrs(format!(
            "IAU:{} WKT names a different planetary body than {}",
            authority.1, expected.name
        )));
    }
    if crate::geo::projections::iau_body(code).is_some() {
        let projected = crate::geo::projections::iau_projection(expected, code).is_some();
        let geographic = crate::geo::projections::iau_geographic(expected, code);
        let wkt_projected =
            upper.trim_start().starts_with("PROJCRS[") || upper.trim_start().starts_with("PROJCS[");
        let wkt_geographic =
            upper.trim_start().starts_with("GEOGCRS[") || upper.trim_start().starts_with("GEOGCS[");
        if (projected && !wkt_projected) || (geographic && !wkt_geographic) {
            return Err(GisError::InvalidCrs(format!(
                "IAU:{} WKT kind does not match the authority code",
                authority.1
            )));
        }
    }
    Ok(())
}

fn looks_like_wkt(value: &str) -> bool {
    let upper = value.trim_start().to_ascii_uppercase();
    ["GEOGCRS[", "PROJCRS[", "GEOGCS[", "PROJCS["]
        .iter()
        .any(|prefix| upper.starts_with(prefix))
}

fn uniform_nodata(values: &[Option<f64>]) -> Option<f64> {
    let first = values.iter().copied().flatten().next()?;
    if values.iter().all(|value| {
        value.is_some_and(|candidate| candidate == first || (candidate.is_nan() && first.is_nan()))
    }) {
        Some(first)
    } else {
        None
    }
}

fn interleave_bhw<T: Copy>(data: &[T], array: &RasterArray) -> Vec<T> {
    let pixels = array.height * array.width;
    let mut interleaved = Vec::with_capacity(data.len());
    for pixel in 0..pixels {
        for band in 0..array.bands {
            interleaved.push(data[band * pixels + pixel]);
        }
    }
    interleaved
}

fn dtype_bits(dtype: RasterDType) -> u16 {
    match dtype {
        RasterDType::UInt8 => 8,
        RasterDType::Int16 | RasterDType::UInt16 => 16,
        RasterDType::Int32 | RasterDType::UInt32 | RasterDType::Float32 => 32,
        RasterDType::Float64 => 64,
    }
}

fn dtype_sample_format(dtype: RasterDType) -> u16 {
    match dtype {
        RasterDType::UInt8 | RasterDType::UInt16 | RasterDType::UInt32 => 1,
        RasterDType::Int16 | RasterDType::Int32 => 2,
        RasterDType::Float32 | RasterDType::Float64 => 3,
    }
}

fn validate_post_write(
    info: &RasterInfo,
    array: &RasterArray,
    options: &WriteRasterOptions,
    expected_path: &Path,
) -> GisResult<()> {
    if Path::new(&info.path) != expected_path {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write path is {:?}, expected {}",
            info.path,
            expected_path.display()
        )));
    }
    if info.driver != "GTiff" {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write driver is {:?}, expected GTiff",
            info.driver
        )));
    }
    if info.width as usize != array.width
        || info.height as usize != array.height
        || info.band_count as usize != array.bands
    {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write dimensions are {}x{}x{}, expected {}x{}x{}",
            info.band_count, info.height, info.width, array.bands, array.height, array.width
        )));
    }
    if info
        .dtype_per_band
        .iter()
        .any(|dtype| dtype != array.dtype().name())
    {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write dtype {:?} does not match {}",
            info.dtype_per_band,
            array.dtype().name()
        )));
    }
    if info.dtype_per_band.len() != array.bands {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write dtype count {} does not match band count {}",
            info.dtype_per_band.len(),
            array.bands
        )));
    }
    if let Some(expected) = &options.crs {
        if !expected.equivalent_to_info(info) {
            return Err(GisError::PostWriteValidationFailed(
                "post-write CRS differs from requested CRS".to_string(),
            ));
        }
    }
    if let Some(expected) = options.transform {
        let Some(actual) = info.transform else {
            return Err(GisError::PostWriteValidationFailed(
                "post-write file is missing transform".to_string(),
            ));
        };
        let actual =
            AffineTransform::new([actual.0, actual.1, actual.2, actual.3, actual.4, actual.5])?;
        if !affine_eq(expected, actual) {
            return Err(GisError::PostWriteValidationFailed(
                "post-write transform differs from requested transform".to_string(),
            ));
        }
    }
    if options.nodata.iter().any(Option::is_some)
        && !nodata_values_eq(&options.nodata, &info.nodata_per_band)
    {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write nodata {:?} does not match requested {:?}",
            info.nodata_per_band, options.nodata
        )));
    }
    let expected_compression = compression_choice_name(options.creation_options.compression);
    if info.compression.as_deref() != Some(expected_compression) {
        return Err(GisError::PostWriteValidationFailed(format!(
            "post-write compression {:?} does not match requested {expected_compression}",
            info.compression
        )));
    }
    Ok(())
}

fn nodata_values_eq(left: &[Option<f64>], right: &[Option<f64>]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (Some(left), Some(right)) => left == right || (left.is_nan() && right.is_nan()),
                (None, None) => true,
                _ => false,
            })
}

fn compression_choice_name(compression: CompressionChoice) -> &'static str {
    match compression {
        CompressionChoice::None => "NONE",
        CompressionChoice::Lzw => "LZW",
        CompressionChoice::Deflate => "DEFLATE",
        CompressionChoice::Packbits => "PACKBITS",
    }
}

fn compression_choice_from_name(compression: &str) -> GisResult<CompressionChoice> {
    match compression.to_ascii_uppercase().as_str() {
        "NONE" | "UNCOMPRESSED" => Ok(CompressionChoice::None),
        "LZW" => Ok(CompressionChoice::Lzw),
        "DEFLATE" | "ZLIB" => Ok(CompressionChoice::Deflate),
        "PACKBITS" => Ok(CompressionChoice::Packbits),
        other => Err(GisError::UnsupportedCreationOption(format!(
            "unsupported like_* compression value {other:?}"
        ))),
    }
}

fn parse_bool(value: &str) -> GisResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "false" | "0" | "no" | "off" => Ok(false),
        "true" | "1" | "yes" | "on" => Ok(true),
        other => Err(GisError::InvalidArgument(format!(
            "invalid boolean creation option value {other:?}"
        ))),
    }
}

fn temporary_sibling_path(path: &Path) -> PathBuf {
    let file_stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("raster");
    let extension = path
        .extension()
        .and_then(|name| name.to_str())
        .unwrap_or("tif");
    let tmp_name = format!(".{file_stem}.tmp-{}.{}", std::process::id(), extension);
    path.with_file_name(tmp_name)
}

fn replace_file(source: &Path, target: &Path) -> GisResult<()> {
    replace_file_impl(source, target)
}

#[cfg(windows)]
fn replace_file_impl(source: &Path, target: &Path) -> GisResult<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source_wide = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let target_wide = target
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let ok = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            target_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if ok == 0 {
        Err(GisError::WriteFailed(
            std::io::Error::last_os_error().to_string(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_file_impl(source: &Path, target: &Path) -> GisResult<()> {
    fs::rename(source, target).map_err(|err| GisError::WriteFailed(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_key(directory: &[u16], key: u16, value: u16) -> bool {
        directory[4..]
            .chunks_exact(4)
            .any(|entry| entry[0] == key && entry[1] == 0 && entry[3] == value)
    }

    #[test]
    fn iau_geokeys_declare_angular_and_projected_units() {
        let geographic = CrsSpec::from_string("IAU:30100".to_string()).unwrap();
        let projected = CrsSpec::from_string("IAU:30115".to_string()).unwrap();
        let polar = CrsSpec::from_string("IAU:30130".to_string()).unwrap();
        let geographic_tags = geokeys_for_crs(&geographic).unwrap().unwrap();
        let projected_tags = geokeys_for_crs(&projected).unwrap().unwrap();
        let polar_tags = geokeys_for_crs(&polar).unwrap().unwrap();
        assert!(contains_key(&geographic_tags.directory, 2054, 9102));
        assert!(contains_key(&geographic_tags.directory, 2056, 32767));
        assert!(contains_key(&projected_tags.directory, 3075, 17));
        assert!(contains_key(&projected_tags.directory, 3076, 9001));
        assert!(polar_tags.directory[4..]
            .chunks_exact(4)
            .any(|entry| entry[0] == 3095 && entry[1] == 34736));
        assert_eq!(geographic_tags.doubles, [1_737_400.0, 1_737_400.0]);
        assert_eq!(
            projected_tags.doubles,
            [1_737_400.0, 1_737_400.0, 0.0, 0.0, 0.0, 180.0, 0.0]
        );
    }

    #[test]
    fn iau_ascii_keeps_authority_before_optional_wkt() {
        let mut supported = CrsSpec::from_string("IAU:30115".to_string()).unwrap();
        supported.wkt = Some("PROJCRS[\"Moon\"]".to_string());
        assert_eq!(
            crs_ascii_for_crs(&supported).as_deref(),
            Some("IAU:30115|PROJCRS[\"Moon\"]|")
        );
        let tags = geokeys_for_crs(&supported).unwrap().unwrap();
        let entries = tags.directory[4..].chunks_exact(4).collect::<Vec<_>>();
        assert!(entries
            .iter()
            .any(|entry| entry[0] == 2049 && entry[2] == 10 && entry[3] == 0));
        assert!(entries
            .iter()
            .any(|entry| entry[0] == 1026 && entry[3] == 10));

        let unknown = CrsSpec {
            authority: Some(("IAU".to_string(), "30199".to_string())),
            wkt: None,
            projection: None,
        };
        assert!(geokeys_for_crs(&unknown).is_err());
        let unknown_with_wkt = CrsSpec {
            wkt: Some("PROJCRS[\"Unknown lunar projection\"]".to_string()),
            ..unknown
        };
        let tags = geokeys_for_crs(&unknown_with_wkt).unwrap().unwrap();
        assert!(contains_key(&tags.directory, 3072, 32767));
        assert!(tags.directory[4..]
            .chunks_exact(4)
            .any(|entry| entry[0] == 3073 && entry[1] == 34737 && entry[3] == 0));
    }
}
