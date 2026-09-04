use super::*;

#[pymethods]
impl TerrainRenderParams {
    #[new]
    #[pyo3(signature = (params))]
    pub fn new(py: Python<'_>, params: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_python_params(py, params)
    }

    #[getter]
    pub fn size_px(&self) -> (u32, u32) {
        self.size_px
    }

    #[getter]
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    #[getter]
    pub fn msaa_samples(&self) -> u32 {
        self.msaa_samples
    }

    #[getter]
    pub fn z_scale(&self) -> f32 {
        self.z_scale
    }

    #[getter]
    pub fn cam_target(&self) -> [f32; 3] {
        self.cam_target
    }

    #[getter]
    pub fn cam_radius(&self) -> f32 {
        self.cam_radius
    }

    #[getter]
    pub fn cam_phi_deg(&self) -> f32 {
        self.cam_phi_deg
    }

    #[getter]
    pub fn cam_theta_deg(&self) -> f32 {
        self.cam_theta_deg
    }

    #[getter]
    pub fn cam_gamma_deg(&self) -> f32 {
        self.cam_gamma_deg
    }

    #[getter]
    pub fn fov_y_deg(&self) -> f32 {
        self.fov_y_deg
    }

    #[getter]
    pub fn clip(&self) -> (f32, f32) {
        self.clip
    }

    #[getter]
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    #[getter]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    #[getter]
    pub fn albedo_mode(&self) -> &str {
        &self.albedo_mode
    }

    #[getter]
    pub fn colormap_strength(&self) -> f32 {
        self.colormap_strength
    }

    #[getter]
    pub fn hue_variation_strength(&self) -> f32 {
        self.hue_variation_strength
    }

    /// P5: Get AO weight (0.0 = no AO, 1.0 = full AO)
    #[getter]
    pub fn ao_weight(&self) -> f32 {
        self.ao_weight
    }

    #[getter]
    pub fn height_curve_mode(&self) -> &str {
        &self.height_curve_mode
    }

    #[getter]
    pub fn height_curve_strength(&self) -> f32 {
        self.height_curve_strength
    }

    #[getter]
    pub fn height_curve_power(&self) -> f32 {
        self.height_curve_power
    }

    /// P5-L: Lambert contrast parameter [0,1] for gradient enhancement
    #[getter]
    pub fn lambert_contrast(&self) -> f32 {
        self.lambert_contrast
    }

    /// P6.1: Use Rgba8UnormSrgb for colormap texture (correct color space sampling)
    #[getter]
    pub fn colormap_srgb(&self) -> bool {
        self.colormap_srgb
    }

    /// P6.1: Use exact linear_to_srgb() instead of pow-gamma for output encoding
    #[getter]
    pub fn output_srgb_eotf(&self) -> bool {
        self.output_srgb_eotf
    }

    /// P7: Camera projection mode ("screen" = fullscreen triangle, "mesh" = perspective grid)
    #[getter]
    pub fn camera_mode(&self) -> &str {
        &self.camera_mode
    }

    #[getter]
    pub fn culling(&self) -> &str {
        &self.culling
    }

    #[getter]
    pub fn shading(&self) -> &str {
        &self.shading
    }

    #[getter]
    pub fn vt_store(&self) -> Option<String> {
        self.vt_store_path.clone()
    }

    #[getter]
    pub fn prefetch_horizon_ms(&self) -> f32 {
        self.prefetch_horizon_ms
    }

    #[getter]
    pub fn vt_upload_budget_bytes(&self) -> u64 {
        self.vt_upload_budget_bytes
    }

    /// P7: Debug mode for projection probes (0=normal, 40=view-depth, 41=NDC depth, 42=view-pos XYZ)
    #[getter]
    pub fn debug_mode(&self) -> u32 {
        self.debug_mode
    }

    /// M1: Accumulation AA sample count (1 = no AA, 16/64/256 typical for offline)
    #[getter]
    pub fn aa_samples(&self) -> u32 {
        self.aa_samples
    }

    /// M1: Accumulation AA seed for deterministic jitter (None = default sequence)
    #[getter]
    pub fn aa_seed(&self) -> Option<u64> {
        self.aa_seed
    }

    #[getter]
    pub fn terrain_data_revision(&self) -> Option<u64> {
        self.terrain_data_revision
    }

    #[getter]
    pub fn height_curve_lut(&self) -> Option<Vec<f32>> {
        self.height_curve_lut
            .as_ref()
            .map(|lut| lut.as_ref().clone())
    }

    #[getter]
    pub fn overlays(&self) -> Vec<Py<crate::core::overlay_layer::OverlayLayer>> {
        self.overlays.clone()
    }

    #[getter]
    pub fn light<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.light.clone_ref(py)
    }

    #[getter]
    pub fn ibl<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.ibl.clone_ref(py)
    }

    #[getter]
    pub fn shadows<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.shadows.clone_ref(py)
    }

    #[getter]
    pub fn triplanar<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.triplanar.clone_ref(py)
    }

    #[getter]
    pub fn pom<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.pom.clone_ref(py)
    }

    #[getter]
    pub fn lod<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.lod.clone_ref(py)
    }

    #[getter]
    pub fn sampling<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.sampling.clone_ref(py)
    }

    #[getter]
    pub fn clamp<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.clamp.clone_ref(py)
    }

    #[getter]
    pub fn python_object<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.python_object.clone_ref(py)
    }

    #[getter]
    pub fn material_map_paths(&self) -> std::collections::BTreeMap<String, String> {
        let mut paths = std::collections::BTreeMap::new();
        if let Some(path) = self.decoded.materials.normal_path.as_ref() {
            paths.insert("normal".to_string(), path.clone());
        }
        if let Some(path) = self.decoded.materials.roughness_path.as_ref() {
            paths.insert("roughness".to_string(), path.clone());
        }
        if let Some(path) = self.decoded.materials.mask_path.as_ref() {
            paths.insert("mask".to_string(), path.clone());
        }
        paths
    }

    pub fn screen_space_settings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let settings = &self.decoded.screen_space;
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("enabled", settings.enabled)?;
        dict.set_item("ssao_enabled", settings.ssao_enabled)?;
        dict.set_item("ssao_radius", settings.ssao_radius)?;
        dict.set_item("ssao_intensity", settings.ssao_intensity)?;
        dict.set_item("ssgi_enabled", settings.ssgi_enabled)?;
        dict.set_item("ssgi_intensity", settings.ssgi_intensity)?;
        dict.set_item("ssr_enabled", settings.ssr_enabled)?;
        dict.set_item("ssr_intensity", settings.ssr_intensity)?;
        dict.set_item("taa_enabled", settings.taa_enabled)?;
        dict.set_item("temporal_alpha", settings.temporal_alpha)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainRenderParams(size_px=({},{}) , overlays={}, msaa_samples={})",
            self.size_px.0,
            self.size_px.1,
            self.overlays.len(),
            self.msaa_samples
        )
    }
}
