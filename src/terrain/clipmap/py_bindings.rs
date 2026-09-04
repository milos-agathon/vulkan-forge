//! P2.1/M5: PyO3 bindings for clipmap terrain system.

use super::level::{calculate_triangle_reduction, clipmap_generate};
use super::vertex::ClipmapVertex;
use super::ClipmapConfig;
use glam::Vec2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-exposed clipmap configuration.
#[pyclass(name = "ClipmapConfig")]
#[derive(Clone)]
pub struct PyClipmapConfig {
    inner: ClipmapConfig,
}

#[pymethods]
impl PyClipmapConfig {
    #[new]
    #[pyo3(signature = (ring_count=4, ring_resolution=64, center_resolution=None, skirt_depth=10.0, morph_range=0.3))]
    fn new(
        ring_count: u32,
        ring_resolution: u32,
        center_resolution: Option<u32>,
        skirt_depth: f32,
        morph_range: f32,
    ) -> PyResult<Self> {
        let center_resolution = center_resolution.unwrap_or(ring_resolution);
        if ring_resolution == 0 || center_resolution == 0 {
            return Err(PyValueError::new_err(
                "ring_resolution and center_resolution must be positive",
            ));
        }
        if !skirt_depth.is_finite() || !morph_range.is_finite() {
            return Err(PyValueError::new_err(
                "skirt_depth and morph_range must be finite",
            ));
        }
        Ok(Self {
            inner: ClipmapConfig {
                ring_count,
                ring_resolution,
                center_resolution,
                skirt_depth,
                morph_range: morph_range.clamp(0.0, 1.0),
            },
        })
    }

    #[getter]
    fn ring_count(&self) -> u32 {
        self.inner.ring_count
    }

    #[getter]
    fn ring_resolution(&self) -> u32 {
        self.inner.ring_resolution
    }

    #[getter]
    fn center_resolution(&self) -> u32 {
        self.inner.center_resolution
    }

    #[getter]
    fn skirt_depth(&self) -> f32 {
        self.inner.skirt_depth
    }

    #[getter]
    fn morph_range(&self) -> f32 {
        self.inner.morph_range
    }

    fn __repr__(&self) -> String {
        format!(
            "ClipmapConfig(ring_count={}, ring_resolution={}, skirt_depth={:.1}, morph_range={:.2})",
            self.inner.ring_count,
            self.inner.ring_resolution,
            self.inner.skirt_depth,
            self.inner.morph_range
        )
    }
}

/// Python-exposed clipmap mesh result.
#[pyclass(name = "ClipmapMesh")]
pub struct PyClipmapMesh {
    vertices: Vec<ClipmapVertex>,
    indices: Vec<u32>,
    triangle_count: u32,
    ring_count: u32,
    full_res_triangles: u32,
}

#[pymethods]
impl PyClipmapMesh {
    #[getter]
    fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    #[getter]
    fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    #[getter]
    fn triangle_count(&self) -> u32 {
        self.triangle_count
    }

    #[getter]
    fn triangle_reduction_percent(&self) -> f32 {
        if self.full_res_triangles == 0 {
            return 0.0;
        }
        let reduction = (self.full_res_triangles as f32 - self.triangle_count as f32)
            / self.full_res_triangles as f32;
        (reduction * 100.0).max(0.0)
    }

    #[getter]
    fn rings_count(&self) -> u32 {
        self.ring_count
    }

    /// Get vertex positions as numpy array (N, 2).
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data: Vec<Vec<f32>> = self
            .vertices
            .iter()
            .map(|v| vec![v.position[0], v.position[1]])
            .collect();
        Ok(PyArray2::from_vec2_bound(py, &data)?)
    }

    /// Get UV coordinates as numpy array (N, 2).
    fn uvs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data: Vec<Vec<f32>> = self
            .vertices
            .iter()
            .map(|v| vec![v.uv[0], v.uv[1]])
            .collect();
        Ok(PyArray2::from_vec2_bound(py, &data)?)
    }

    /// Get morph data as numpy array (N, 2): [morph_weight, ring_index].
    fn morph_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data: Vec<Vec<f32>> = self
            .vertices
            .iter()
            .map(|v| vec![v.morph_data[0], v.morph_data[1]])
            .collect();
        Ok(PyArray2::from_vec2_bound(py, &data)?)
    }

    /// Get indices as numpy array (M,).
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_vec_bound(py, self.indices.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "ClipmapMesh(vertices={}, triangles={}, reduction={:.1}%)",
            self.vertex_count(),
            self.triangle_count,
            self.triangle_reduction_percent()
        )
    }
}

/// Generate a clipmap mesh from configuration.
#[pyfunction]
#[pyo3(signature = (config, center, terrain_extent))]
pub fn clipmap_generate_py(
    config: &PyClipmapConfig,
    center: (f32, f32),
    terrain_extent: f32,
) -> PyResult<PyClipmapMesh> {
    if !center.0.is_finite() || !center.1.is_finite() {
        return Err(PyValueError::new_err("clipmap center must be finite"));
    }
    if !terrain_extent.is_finite() || terrain_extent <= 0.0 {
        return Err(PyValueError::new_err(
            "terrain_extent must be finite and positive",
        ));
    }
    let center_vec = Vec2::new(center.0, center.1);
    let mesh = clipmap_generate(&config.inner, center_vec, terrain_extent);

    // Calculate full-res triangle count for comparison
    let full_res_triangles = super::level::full_resolution_triangle_count(&config.inner);

    Ok(PyClipmapMesh {
        vertices: mesh.vertices,
        indices: mesh.indices,
        triangle_count: mesh.triangle_count,
        ring_count: config.inner.ring_count,
        full_res_triangles,
    })
}

/// Calculate triangle reduction percentage.
#[pyfunction]
pub fn calculate_triangle_reduction_py(full_res_triangles: u32, clipmap_triangles: u32) -> f32 {
    calculate_triangle_reduction(full_res_triangles, clipmap_triangles) * 100.0
}

/// Register clipmap Python bindings.
pub fn register_clipmap_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClipmapConfig>()?;
    m.add_class::<PyClipmapMesh>()?;
    m.add_function(wrap_pyfunction!(clipmap_generate_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_triangle_reduction_py, m)?)?;
    Ok(())
}
