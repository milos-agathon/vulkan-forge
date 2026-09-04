use super::super::*;
use pyo3::exceptions::PyKeyError;

use crate::core::atmosphere::{
    AtmosphereLutHandle as CoreAtmosphereLutHandle, LutDimensions,
    AERIAL_TRANSMITTANCE_LUT_SEMANTICS,
};

const REPORT_KEYS: [&str; 20] = [
    "turbidity",
    "ozone_du",
    "mie_g",
    "ground_albedo",
    "scattering_orders",
    "wavelength_count",
    "wavelengths_nm",
    "storage_format",
    "scattering_lut_semantics",
    "aerial_lut_semantics",
    "precomputed",
    "precomputed_turbidity_bracket",
    "dimensions",
    "byte_size",
    "deterministic_sha256",
    "order_deltas",
    "transmittance_rgba",
    "single_scattering_rgba",
    "multiple_scattering_rgba",
    "aerial_perspective_rgba",
];

/// Immutable native handoff for an exact tracked AETHER LUT payload.
#[pyclass(module = "forge3d._forge3d", name = "AtmosphereLutHandle", frozen)]
#[derive(Clone)]
pub struct PyAtmosphereLutHandle {
    handle: CoreAtmosphereLutHandle,
}

impl PyAtmosphereLutHandle {
    pub(crate) fn new(handle: CoreAtmosphereLutHandle) -> Self {
        Self { handle }
    }

    pub(crate) fn core_handle(&self) -> &CoreAtmosphereLutHandle {
        &self.handle
    }

    fn dimensions_dict(py: Python<'_>, dimensions: LutDimensions) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "transmittance",
            [
                dimensions.transmittance_mu,
                dimensions.transmittance_height,
                1,
            ],
        )?;
        dict.set_item(
            "single_scattering",
            [
                dimensions.scattering_mu_view,
                dimensions.scattering_mu_sun,
                dimensions.scattering_height * dimensions.scattering_nu,
            ],
        )?;
        dict.set_item(
            "multiple_scattering",
            [
                dimensions.scattering_mu_view,
                dimensions.scattering_mu_sun,
                dimensions.scattering_height * dimensions.scattering_nu,
            ],
        )?;
        dict.set_item(
            "aerial_perspective",
            [
                dimensions.aerial_distance,
                dimensions.aerial_mu_view,
                dimensions.aerial_height,
            ],
        )?;
        Ok(dict.into_py(py))
    }

    fn value_for_key(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        let luts = self.handle.luts();
        let config = self.handle.config();
        match key {
            "turbidity" => Ok(config.turbidity.into_py(py)),
            "ozone_du" => Ok(config.ozone_du.into_py(py)),
            "mie_g" => Ok(config.mie_g.into_py(py)),
            "ground_albedo" => Ok(config.ground_albedo.into_py(py)),
            "scattering_orders" => Ok(config.scattering_orders.into_py(py)),
            "wavelength_count" => Ok(luts.metadata.wavelengths_nm.len().into_py(py)),
            "wavelengths_nm" => Ok(luts.metadata.wavelengths_nm.to_vec().into_py(py)),
            "storage_format" => Ok(luts.metadata.storage_format.into_py(py)),
            "scattering_lut_semantics" => Ok(luts.metadata.scattering_lut_semantics.into_py(py)),
            "aerial_lut_semantics" => Ok(AERIAL_TRANSMITTANCE_LUT_SEMANTICS.into_py(py)),
            "precomputed" => Ok(luts.metadata.precomputed.into_py(py)),
            "precomputed_turbidity_bracket" => {
                Ok(luts.metadata.precomputed_turbidity_bracket.into_py(py))
            }
            "dimensions" => Self::dimensions_dict(py, luts.metadata.dimensions),
            "byte_size" => Ok(luts.byte_size().into_py(py)),
            "deterministic_sha256" => Ok(self.handle.deterministic_sha256_hex().into_py(py)),
            "order_deltas" => Ok(luts.order_deltas.clone().into_py(py)),
            "transmittance_rgba" => {
                Ok(PyArray1::from_vec_bound(py, luts.transmittance.rgba_f32()).into_py(py))
            }
            "single_scattering_rgba" => {
                Ok(PyArray1::from_vec_bound(py, luts.single_scattering.rgba_f32()).into_py(py))
            }
            "multiple_scattering_rgba" => {
                Ok(PyArray1::from_vec_bound(py, luts.multiple_scattering.rgba_f32()).into_py(py))
            }
            "aerial_perspective_rgba" => {
                Ok(PyArray1::from_vec_bound(py, luts.aerial_perspective.rgba_f32()).into_py(py))
            }
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn as_dict_impl(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for key in REPORT_KEYS {
            dict.set_item(key, self.value_for_key(py, key)?)?;
        }
        Ok(dict.into_py(py))
    }
}

#[pymethods]
impl PyAtmosphereLutHandle {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "AtmosphereLutHandle objects are returned by atmosphere_bake_luts()",
        ))
    }

    #[getter]
    fn turbidity(&self) -> f32 {
        self.handle.config().turbidity
    }

    #[getter]
    fn ozone_du(&self) -> f32 {
        self.handle.config().ozone_du
    }

    #[getter]
    fn mie_g(&self) -> f32 {
        self.handle.config().mie_g
    }

    #[getter]
    fn ground_albedo(&self) -> f32 {
        self.handle.config().ground_albedo
    }

    #[getter]
    fn scattering_orders(&self) -> u32 {
        self.handle.config().scattering_orders
    }

    #[getter]
    fn precomputed(&self) -> bool {
        self.handle.luts().metadata.precomputed
    }

    #[getter]
    fn byte_size(&self) -> u64 {
        self.handle.luts().byte_size()
    }

    #[getter]
    fn deterministic_sha256(&self) -> String {
        self.handle.deterministic_sha256_hex()
    }

    /// Optional froxel semantics: RGB is zero; alpha is mean segment T.
    #[getter]
    fn aerial_lut_semantics(&self) -> &'static str {
        AERIAL_TRANSMITTANCE_LUT_SEMANTICS
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.value_for_key(py, key)
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        key.extract::<String>()
            .ok()
            .is_some_and(|key| REPORT_KEYS.contains(&key.as_str()))
    }

    fn __len__(&self) -> usize {
        REPORT_KEYS.len()
    }

    fn keys(&self) -> Vec<&'static str> {
        REPORT_KEYS.to_vec()
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.as_dict_impl(py)
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        let config = self.handle.config();
        format!(
            "AtmosphereLutHandle(turbidity={}, ozone_du={}, mie_g={}, ground_albedo={}, precomputed={}, sha256='{}')",
            config.turbidity,
            config.ozone_du,
            config.mie_g,
            config.ground_albedo,
            self.handle.luts().metadata.precomputed,
            self.handle.deterministic_sha256_hex(),
        )
    }
}
