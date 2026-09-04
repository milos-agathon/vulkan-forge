use super::*;

impl Scene {
    pub(crate) fn ensure_legacy_scene_atmosphere_unconfigured(&self) -> PyResult<()> {
        if self.atmosphere_state.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "AETHER is configured on Scene, but Scene has no spectral-atmosphere ".to_owned()
                    + "render consumer. Use TerrainRenderer with "
                    + "SkySettings(enabled=True, model='aether'), or call "
                    + "Scene.clear_atmosphere() before rendering. The render was blocked "
                    + "instead of silently using the legacy RGB sky/fog path.",
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    /// Resolve validated AETHER settings against the shipped LUT bank.
    ///
    /// This configures provenance only. Active AETHER rendering is provided by
    /// `TerrainRenderer` with `SkySettings(model="aether")`; Scene's legacy
    /// pixel entry points fail closed while this state is configured.
    #[pyo3(
        signature = (turbidity = 2.0, ozone_du = 300.0, mie_g = 0.8),
        text_signature = "($self, turbidity=2.0, ozone_du=300.0, mie_g=0.8)"
    )]
    pub fn set_atmosphere(&mut self, turbidity: f32, ozone_du: f32, mie_g: f32) -> PyResult<()> {
        let config = crate::core::atmosphere::AtmosphereConfig {
            turbidity,
            ozone_du,
            mie_g,
            ..Default::default()
        };
        config
            .validate()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;

        let luts = crate::core::atmosphere::load_precomputed_atmosphere_luts(config.clone())
            .map_err(|error| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Scene.set_atmosphere could not resolve the shipped AETHER LUT bank: \
                     {error}. Custom ozone/mie inputs are accepted by the explicit \
                     atmosphere_bake_luts offline API, but Scene state can reference only \
                     shipped transport tables; no nearby or legacy LUT was substituted."
                ))
            })?;
        let deterministic_sha256 = luts
            .deterministic_sha256()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        self.atmosphere_state = Some(SceneAtmosphereState {
            config,
            precomputed_turbidity_bracket: luts.metadata.precomputed_turbidity_bracket,
            byte_size: luts.byte_size(),
            deterministic_sha256,
        });
        Ok(())
    }

    /// Clear the configured AETHER state and restore normal legacy Scene renders.
    #[pyo3(text_signature = "($self)")]
    pub fn clear_atmosphere(&mut self) {
        self.atmosphere_state = None;
    }

    /// Return validated AETHER provenance without claiming this Scene is active.
    #[pyo3(text_signature = "($self)")]
    pub fn get_atmosphere_settings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("configured", self.atmosphere_state.is_some())?;
        dict.set_item("active", false)?;
        dict.set_item(
            "active_render_path",
            "TerrainRenderer with SkySettings(enabled=True, model='aether')",
        )?;
        if let Some(state) = &self.atmosphere_state {
            dict.set_item("turbidity", state.config.turbidity)?;
            dict.set_item("ozone_du", state.config.ozone_du)?;
            dict.set_item("mie_g", state.config.mie_g)?;
            dict.set_item("precomputed", true)?;
            dict.set_item(
                "precomputed_turbidity_bracket",
                state.precomputed_turbidity_bracket,
            )?;
            dict.set_item("lut_byte_size", state.byte_size)?;
            dict.set_item("deterministic_sha256", &state.deterministic_sha256)?;
        }
        Ok(dict.into())
    }
}
