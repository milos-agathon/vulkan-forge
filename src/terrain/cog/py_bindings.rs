//! P3: PyO3 bindings for COG streaming.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::cog_reader::CogHeightReader;

/// Python wrapper for COG dataset.
#[pyclass(module = "forge3d._forge3d", name = "CogDataset")]
pub struct PyCogDataset {
    reader: Arc<CogHeightReader>,
    _runtime: tokio::runtime::Runtime,
    url: String,
}

impl PyCogDataset {
    pub(crate) fn reader(&self) -> Arc<CogHeightReader> {
        self.reader.clone()
    }
}

#[pymethods]
impl PyCogDataset {
    /// Open a COG dataset from a URL.
    ///
    /// Args:
    ///     url: HTTP(S) URL or file:// path to COG file
    ///     cache_size_mb: Tile cache memory budget in MB (default: 256)
    ///     cache_dir: Optional on-disk range cache directory
    ///     cache_budget_mb: Byte/disk range cache budget in MB (default: cache_size_mb)
    #[new]
    #[pyo3(signature = (url, cache_size_mb=256, cache_dir=None, cache_budget_mb=None))]
    pub fn new(
        url: &str,
        cache_size_mb: u32,
        cache_dir: Option<String>,
        cache_budget_mb: Option<u32>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create tokio runtime: {}", e))
            })?;

        let handle = runtime.handle().clone();
        let reader = runtime
            .block_on(async {
                CogHeightReader::new_with_runtime_and_cache_options(
                    url,
                    cache_size_mb,
                    handle,
                    cache_dir.map(PathBuf::from),
                    cache_budget_mb.unwrap_or(cache_size_mb),
                )
                .await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open COG: {:?}", e)))?;

        Ok(Self {
            reader: Arc::new(reader),
            _runtime: runtime,
            url: url.to_string(),
        })
    }

    /// Get geographic bounds (minx, miny, maxx, maxy).
    #[getter]
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        self.reader.bounds()
    }

    /// Get number of overview levels.
    #[getter]
    pub fn overview_count(&self) -> usize {
        self.reader.overview_count()
    }

    /// Get the URL of this dataset.
    #[getter]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Read a single tile at specified coordinates and LOD.
    ///
    /// Args:
    ///     x: Tile X coordinate
    ///     y: Tile Y coordinate  
    ///     lod: Level of detail (0 = full resolution)
    ///
    /// Returns:
    ///     2D numpy array of float32 heights
    #[pyo3(signature = (x, y, lod=0))]
    pub fn read_tile<'py>(
        &self,
        py: Python<'py>,
        x: u32,
        y: u32,
        lod: u32,
    ) -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f32>>> {
        let heights = self
            ._runtime
            .block_on(self.reader.read_tile_async(x, y, lod))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read tile: {:?}", e)))?;

        let header = self.reader.header();
        let ifd = header
            .select_ifd_for_lod(lod)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to select IFD: {:?}", e)))?;
        let tile_width = ifd.tile_width as usize;
        let tile_height = ifd.tile_height as usize;

        let expected_len = tile_width * tile_height;
        if heights.len() != expected_len {
            return Err(PyRuntimeError::new_err(format!(
                "Unexpected tile size: got {}, expected {}",
                heights.len(),
                expected_len
            )));
        }

        let arr =
            numpy::PyArray2::from_vec2_bound(py, &vec_to_2d(&heights, tile_height, tile_width))
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create numpy array: {}", e))
                })?;

        Ok(arr)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> HashMap<String, f64> {
        let stats = self.reader.cache_stats();
        let mut map = HashMap::new();
        map.insert("cache_hits".to_string(), stats.hits as f64);
        map.insert("cache_misses".to_string(), stats.misses as f64);
        map.insert("cache_evictions".to_string(), stats.evictions as f64);
        map.insert(
            "memory_used_bytes".to_string(),
            stats.memory_used_bytes as f64,
        );
        map.insert(
            "memory_budget_bytes".to_string(),
            stats.memory_budget_bytes as f64,
        );
        map.insert(
            "byte_cache_used_bytes".to_string(),
            stats.byte_cache_used_bytes as f64,
        );
        map.insert(
            "byte_cache_budget_bytes".to_string(),
            stats.byte_cache_budget_bytes as f64,
        );
        map.insert(
            "disk_cache_used_bytes".to_string(),
            stats.disk_cache_used_bytes as f64,
        );
        map.insert(
            "disk_cache_budget_bytes".to_string(),
            stats.disk_cache_budget_bytes as f64,
        );

        let total = stats.hits + stats.misses;
        let hit_rate = if total > 0 {
            stats.hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        map.insert("hit_rate_percent".to_string(), hit_rate);

        map
    }

    /// Get information about a specific IFD/overview level.
    #[pyo3(signature = (level=0))]
    pub fn ifd_info(&self, level: u32) -> PyResult<HashMap<String, u32>> {
        let header = self.reader.header();
        let ifd = header
            .ifds
            .get(level as usize)
            .ok_or_else(|| PyRuntimeError::new_err(format!("IFD level {} not found", level)))?;

        let mut map = HashMap::new();
        map.insert("width".to_string(), ifd.width);
        map.insert("height".to_string(), ifd.height);
        map.insert("tile_width".to_string(), ifd.tile_width);
        map.insert("tile_height".to_string(), ifd.tile_height);
        map.insert("tiles_across".to_string(), ifd.tiles_across);
        map.insert("tiles_down".to_string(), ifd.tiles_down);
        map.insert("bits_per_sample".to_string(), ifd.bits_per_sample as u32);
        map.insert("compression".to_string(), ifd.compression as u32);
        map.insert("tile_count".to_string(), ifd.tile_count() as u32);

        Ok(map)
    }
}

/// Convert flat Vec to 2D Vec for numpy.
fn vec_to_2d(data: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let end = (start + cols).min(data.len());
        if start < data.len() {
            result.push(data[start..end].to_vec());
        } else {
            result.push(vec![0.0; cols]);
        }
    }
    result
}

/// Register COG streaming bindings with the Python module.
pub fn register_cog_bindings(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyCogDataset>()?;
    Ok(())
}
