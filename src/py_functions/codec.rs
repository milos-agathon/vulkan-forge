//! Python boundary for the unconditional F3DZ core.

use super::super::*;
use numpy::{PyArray2, PyReadonlyArray2};
use sha2::{Digest, Sha256};

fn py_codec_error(error: crate::codec::f3dz::F3dzError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
pub(crate) fn encode_bc7_rgba8<'py>(
    py: Python<'py>,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = crate::core::compressed_textures::encode_bc7_rgba8(rgba, width, height)
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
pub(crate) fn decode_bc7_rgba8<'py>(
    py: Python<'py>,
    blocks: &[u8],
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let decoded = crate::core::compressed_textures::decode_bc7_rgba8(blocks, width, height)
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &decoded))
}

#[pyfunction]
pub(crate) fn encode_bc5_rg8<'py>(
    py: Python<'py>,
    rg: &[u8],
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = crate::core::compressed_textures::encode_bc5_rg8(rg, width, height)
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
pub(crate) fn decode_bc5_rg8<'py>(
    py: Python<'py>,
    blocks: &[u8],
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let decoded = crate::core::compressed_textures::decode_bc5_rg8(blocks, width, height)
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &decoded))
}

#[pyfunction]
#[pyo3(signature = (dem, eps, progressive=true))]
pub(crate) fn compress_dem<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    eps: f32,
    progressive: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let array = dem.as_array();
    let height = u32::try_from(array.nrows())
        .map_err(|_| PyValueError::new_err("DEM height exceeds u32"))?;
    let width =
        u32::try_from(array.ncols()).map_err(|_| PyValueError::new_err("DEM width exceeds u32"))?;
    let values: Vec<f32> = array.iter().copied().collect();
    let mut options = crate::codec::f3dz::EncodeOptions::new(eps);
    options.progressive = progressive;
    let encoded =
        crate::codec::f3dz::encode_dem(&values, width, height, &options).map_err(py_codec_error)?;
    Ok(PyBytes::new_bound(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (data))]
pub(crate) fn decompress_dem<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyArray2<f32>>, Py<PyDict>)> {
    let decoded = crate::codec::f3dz::decode_dem(data, None).map_err(py_codec_error)?;
    let (_, entries) = crate::codec::f3dz::format::parse_prefix(data).map_err(py_codec_error)?;
    let array = ndarray::Array2::from_shape_vec(
        (decoded.height as usize, decoded.width as usize),
        decoded.values,
    )
    .map_err(|error| PyRuntimeError::new_err(format!("F3DZ array reshape failed: {error}")))?;
    let info = PyDict::new_bound(py);
    info.set_item("codec", "f3dz/1")?;
    info.set_item("width", decoded.width)?;
    info.set_item("height", decoded.height)?;
    info.set_item("eps", decoded.epsilon)?;
    info.set_item(
        "progressive",
        entries.first().is_some_and(|entry| entry.progressive()),
    )?;
    info.set_item("base_quality", decoded.base_quality)?;
    info.set_item("page_count", entries.len())?;
    info.set_item("height_datum", decoded.height_datum)?;
    Ok((PyArray2::from_owned_array_bound(py, array), info.into()))
}

#[pyfunction]
#[pyo3(signature = (data, source=None))]
pub(crate) fn verify_dem<'py>(
    py: Python<'py>,
    data: &[u8],
    source: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Py<PyDict>> {
    // decode_dem is the authoritative fail-closed structural/CRC verifier.
    let decoded = crate::codec::f3dz::decode_dem(data, None).map_err(py_codec_error)?;
    let (header, entries) =
        crate::codec::f3dz::format::parse_prefix(data).map_err(py_codec_error)?;
    let source = source.map(|array| array.as_array().to_owned());
    if let Some(source) = &source {
        if source.shape() != [decoded.height as usize, decoded.width as usize] {
            return Err(PyValueError::new_err(format!(
                "source shape {:?} does not match F3DZ grid ({}, {})",
                source.shape(),
                decoded.height,
                decoded.width
            )));
        }
    }

    let page_reports = pyo3::types::PyList::empty_bound(py);
    let cpu_page_shas = pyo3::types::PyList::empty_bound(py);
    let mut all_within = true;
    let mut stored_match = true;
    let mut global_max = 0.0f32;
    for (page, entry) in entries.iter().enumerate() {
        let start_x = entry.page_x as usize * header.tile_size as usize;
        let start_y = entry.page_y as usize * header.tile_size as usize;
        let mut max_error = 0.0f32;
        let mut page_within = true;
        if let Some(source) = &source {
            for y in 0..entry.height as usize {
                for x in 0..entry.width as usize {
                    let index = (start_y + y) * decoded.width as usize + start_x + x;
                    let original = source[(start_y + y, start_x + x)];
                    let reconstructed = decoded.values[index];
                    if original.is_nan() || reconstructed.is_nan() {
                        if !(original.is_nan() && reconstructed.is_nan()) {
                            page_within = false;
                            max_error = f32::INFINITY;
                        }
                    } else {
                        let error = (reconstructed - original).abs();
                        max_error = max_error.max(error);
                        page_within &= error <= decoded.epsilon;
                    }
                }
            }
            all_within &= page_within;
            stored_match &= max_error.to_bits() == entry.max_abs_err.to_bits();
            global_max = global_max.max(max_error);
        }
        let report = PyDict::new_bound(py);
        report.set_item("page", page)?;
        report.set_item("page_x", entry.page_x)?;
        report.set_item("page_y", entry.page_y)?;
        report.set_item("crc32", entry.crc32)?;
        report.set_item("stored_max_abs_err", entry.max_abs_err)?;
        report.set_item("measured_max_abs_err", source.as_ref().map(|_| max_error))?;
        report.set_item("within_epsilon", source.as_ref().map(|_| page_within))?;
        report.set_item(
            "stored_error_matches",
            source
                .as_ref()
                .map(|_| max_error.to_bits() == entry.max_abs_err.to_bits()),
        )?;
        let cpu_sha = page_sha256(
            &decoded.values,
            decoded.width as usize,
            start_x,
            start_y,
            entry.width as usize,
            entry.height as usize,
        );
        report.set_item("cpu_sha256", &cpu_sha)?;
        cpu_page_shas.append(cpu_sha)?;
        page_reports.append(report)?;
    }

    let gpu_report = PyDict::new_bound(py);
    gpu_report.set_item("cpu_sha256", values_sha256(&decoded.values))?;
    gpu_report.set_item("cpu_page_sha256", cpu_page_shas)?;
    match crate::codec::f3dz::gpu::decode_dem_gpu(data) {
        Ok(gpu) => {
            let identical = gpu.values.len() == decoded.values.len()
                && gpu
                    .values
                    .iter()
                    .zip(&decoded.values)
                    .all(|(left, right)| left.to_bits() == right.to_bits());
            let gpu_page_shas = pyo3::types::PyList::empty_bound(py);
            for entry in &entries {
                gpu_page_shas.append(page_sha256(
                    &gpu.values,
                    decoded.width as usize,
                    entry.page_x as usize * header.tile_size as usize,
                    entry.page_y as usize * header.tile_size as usize,
                    entry.width as usize,
                    entry.height as usize,
                ))?;
            }
            gpu_report.set_item("available", true)?;
            gpu_report.set_item("adapter", gpu.adapter)?;
            gpu_report.set_item("elapsed_seconds", gpu.elapsed_seconds)?;
            gpu_report.set_item("gpu_sha256", values_sha256(&gpu.values))?;
            gpu_report.set_item("gpu_page_sha256", gpu_page_shas)?;
            gpu_report.set_item("bit_identical", identical)?;
        }
        Err(error) => {
            gpu_report.set_item("available", false)?;
            gpu_report.set_item("error", error.to_string())?;
            gpu_report.set_item("adapter", py.None())?;
            gpu_report.set_item("elapsed_seconds", py.None())?;
            gpu_report.set_item("gpu_sha256", py.None())?;
            gpu_report.set_item("gpu_page_sha256", py.None())?;
            gpu_report.set_item("bit_identical", py.None())?;
        }
    }

    let (ablation_size, baseline_size, base_report) = if let Some(source) = &source {
        let source_values: Vec<f32> = source.iter().copied().collect();
        let mut diagnostic_options = crate::codec::f3dz::EncodeOptions::new(header.epsilon);
        diagnostic_options.progressive = header.progressive();
        diagnostic_options.tile_size = header.tile_size;
        diagnostic_options.height_datum = header.height_datum.clone();
        let ablation_size = if header.base_only() {
            None
        } else {
            let mut options = diagnostic_options.clone();
            options.force_order_zero = true;
            Some(
                crate::codec::f3dz::encode_dem(
                    &source_values,
                    header.width,
                    header.height,
                    &options,
                )
                .map_err(py_codec_error)?
                .len(),
            )
        };
        #[cfg(feature = "cog_streaming")]
        let baseline_size = Some(
            crate::codec::f3dz::encode::flate2_baseline_size(
                &source_values,
                header.width,
                header.height,
                &diagnostic_options,
            )
            .map_err(py_codec_error)?,
        );
        #[cfg(not(feature = "cog_streaming"))]
        let baseline_size: Option<usize> = None;
        let base_report: Option<Py<PyDict>> = if header.progressive() && !header.base_only() {
            let base_stream = crate::codec::f3dz::base_only_stream(data).map_err(py_codec_error)?;
            let base_decoded = crate::codec::f3dz::decode_dem(&base_stream, Some(header.epsilon))
                .map_err(py_codec_error)?;
            let base = PyDict::new_bound(py);
            let mut base_max_error = 0.0f32;
            let mut base_stored_match = true;
            let base_pages = pyo3::types::PyList::empty_bound(py);
            for entry in &entries {
                let start_x = entry.page_x as usize * header.tile_size as usize;
                let start_y = entry.page_y as usize * header.tile_size as usize;
                let measured = page_max_error(
                    source,
                    &base_decoded.values,
                    base_decoded.width as usize,
                    start_x,
                    start_y,
                    entry.width as usize,
                    entry.height as usize,
                );
                base_max_error = base_max_error.max(measured);
                base_stored_match &= measured.to_bits() == entry.base_max_abs_err.to_bits();
                let page = PyDict::new_bound(py);
                page.set_item("page_x", entry.page_x)?;
                page.set_item("page_y", entry.page_y)?;
                page.set_item("stored_max_abs_err", entry.base_max_abs_err)?;
                page.set_item("measured_max_abs_err", measured)?;
                page.set_item(
                    "stored_error_matches",
                    measured.to_bits() == entry.base_max_abs_err.to_bits(),
                )?;
                base_pages.append(page)?;
            }
            base.set_item("stream_size", base_stream.len())?;
            base.set_item("base_quality", base_decoded.base_quality)?;
            base.set_item("max_abs_err", base_max_error)?;
            base.set_item("within_4epsilon", base_max_error <= header.epsilon * 4.0)?;
            base.set_item("stored_errors_match", base_stored_match)?;
            base.set_item("pages", base_pages)?;
            base.set_item("degradation_kind", "base_quality")?;
            base.set_item("degradation_name", "f3dz_unrefined_pages")?;
            Some(base.unbind())
        } else {
            None
        };
        (ablation_size, baseline_size, base_report)
    } else {
        (None, None, None)
    };

    let result = PyDict::new_bound(py);
    result.set_item("valid", true)?;
    result.set_item("codec", "f3dz/1")?;
    result.set_item("crc_ok", true)?;
    result.set_item("header_consistent", true)?;
    result.set_item("width", decoded.width)?;
    result.set_item("height", decoded.height)?;
    result.set_item("eps", decoded.epsilon)?;
    result.set_item("base_quality", decoded.base_quality)?;
    result.set_item("source_checked", source.is_some())?;
    result.set_item("max_abs_err", source.as_ref().map(|_| global_max))?;
    result.set_item("all_within_epsilon", source.as_ref().map(|_| all_within))?;
    result.set_item("stored_errors_match", source.as_ref().map(|_| stored_match))?;
    result.set_item("stream_size", data.len())?;
    result.set_item("ablation_stream_size", ablation_size)?;
    result.set_item("baseline_flate2_stream_size", baseline_size)?;
    result.set_item("gpu", gpu_report)?;
    result.set_item("base", base_report)?;
    result.set_item("pages", page_reports)?;
    Ok(result.into())
}

fn page_max_error(
    source: &ndarray::Array2<f32>,
    decoded: &[f32],
    decoded_width: usize,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
) -> f32 {
    let mut maximum = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let original = source[(start_y + y, start_x + x)];
            let reconstructed = decoded[(start_y + y) * decoded_width + start_x + x];
            if original.is_nan() && reconstructed.is_nan() {
                continue;
            }
            if original.is_nan() || reconstructed.is_nan() {
                return f32::INFINITY;
            }
            maximum = maximum.max((original - reconstructed).abs());
        }
    }
    maximum
}

fn page_sha256(
    values: &[f32],
    full_width: usize,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
) -> String {
    let mut digest = Sha256::new();
    for y in 0..height {
        let start = (start_y + y) * full_width + start_x;
        for value in &values[start..start + width] {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn values_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
