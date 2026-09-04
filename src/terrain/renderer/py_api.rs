use super::*;
use crate::terrain::render_params;
use numpy::PyUntypedArrayMethods;
use pyo3::types::PyDict;
use sha2::{Digest, Sha256};

fn append_content_digest(target: &mut Vec<u8>, domain: &[u8], content: &[u8]) {
    target.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    target.extend_from_slice(domain);
    target.extend_from_slice(&(content.len() as u64).to_le_bytes());
    target.extend_from_slice(&Sha256::digest(content));
}

#[allow(clippy::too_many_arguments)]
fn terrain_cache_options<'py>(
    py: Python<'py>,
    cache: Option<&Bound<'py, PyAny>>,
    scene: &TerrainScene,
    material_set: &crate::render::material_set::MaterialSet,
    env_maps: &crate::lighting::ibl_wrapper::IBL,
    params: &render_params::TerrainRenderParams,
    heightmap: &PyReadonlyArray2<'py, f32>,
    water_mask: Option<&PyReadonlyArray2<'py, f32>>,
    time_seconds: f32,
) -> PyResult<Option<super::anamnesis::TerrainCacheOptions>> {
    let Some(cache) = cache else {
        return Ok(None);
    };
    if matches!(
        params
            .decoded()
            .shadow
            .technique
            .to_ascii_uppercase()
            .as_str(),
        "VSM" | "EVSM" | "MSM"
    ) {
        return Err(PyRuntimeError::new_err(
            "native terrain ANAMNESIS rejects moment-shadow techniques until their moment texture is a declared graph output",
        ));
    }
    let root = cache.extract::<std::path::PathBuf>().or_else(|_| {
        cache
            .call_method0("__fspath__")?
            .extract::<std::path::PathBuf>()
    })?;
    let dataclasses = py.import_bound("dataclasses")?;
    let params_dict = dataclasses
        .getattr("asdict")?
        .call1((params.python_object.bind(py),))?;
    let params_mapping = params_dict.downcast::<PyDict>()?;
    let canonical = py.import_bound("forge3d._canonical_json")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("error_context", "native terrain ANAMNESIS state")?;
    let params_bytes = canonical
        .getattr("canonical_json_bytes")?
        .call((params_dict.clone(),), Some(&kwargs))?
        .extract::<Vec<u8>>()?;
    let shadow_projection = PyDict::new_bound(py);
    for key in [
        "size_px",
        "terrain_span",
        "z_scale",
        "cam_target",
        "cam_radius",
        "cam_phi_deg",
        "cam_theta_deg",
        "fov_y_deg",
        "clip",
        "light",
        "shadows",
        "clamp",
        "height_curve_mode",
        "height_curve_strength",
        "height_curve_power",
        "height_curve_lut",
    ] {
        if let Some(value) = params_mapping.get_item(key)? {
            shadow_projection.set_item(key, value)?;
        }
    }
    kwargs.set_item("error_context", "native terrain ANAMNESIS shadow state")?;
    let mut shadow_bytes = canonical
        .getattr("canonical_json_bytes")?
        .call((shadow_projection,), Some(&kwargs))?
        .extract::<Vec<u8>>()?;

    let mut state = Vec::new();
    state.extend_from_slice(b"forge3d.terrain.native-inputs/v2\0");
    state.extend_from_slice(&(params_bytes.len() as u64).to_le_bytes());
    state.extend_from_slice(&params_bytes);
    let material_bytes = material_set.anamnesis_bytes().map_err(|error| {
        PyRuntimeError::new_err(format!(
            "failed to identify terrain material source: {error}"
        ))
    })?;
    append_content_digest(&mut state, b"material-set", &material_bytes);
    let ibl_bytes = env_maps
        .anamnesis_bytes()
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))?;
    append_content_digest(&mut state, b"ibl", &ibl_bytes);

    let height = heightmap.as_array();
    let mut height_bytes = Vec::new();
    height_bytes.extend_from_slice(&(height.shape()[0] as u64).to_le_bytes());
    height_bytes.extend_from_slice(&(height.shape()[1] as u64).to_le_bytes());
    for value in height.iter() {
        height_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    match water_mask {
        Some(mask) => {
            state.push(1);
            let mask = mask.as_array();
            state.extend_from_slice(&(mask.shape()[0] as u64).to_le_bytes());
            state.extend_from_slice(&(mask.shape()[1] as u64).to_le_bytes());
            let mut digest = Sha256::new();
            for value in mask.iter() {
                digest.update(value.to_bits().to_le_bytes());
            }
            state.extend_from_slice(&digest.finalize());
        }
        None => state.push(0),
    }
    state.extend_from_slice(&time_seconds.to_bits().to_le_bytes());
    for name in [
        "FORGE3D_TERRAIN_SHADOW_DEBUG",
        "VF_COLOR_DEBUG_MODE",
        "VF_ROUGHNESS_MULT",
        "VF_SPEC_AA_ENABLED",
        "VF_SPECAA_SIGMA_SCALE",
        "VF_FORCE_LUT_UNORM",
    ] {
        state.extend_from_slice(&(name.len() as u64).to_le_bytes());
        state.extend_from_slice(name.as_bytes());
        match std::env::var_os(name) {
            Some(value) => {
                state.push(1);
                let value = value.to_string_lossy();
                state.extend_from_slice(&(value.len() as u64).to_le_bytes());
                state.extend_from_slice(value.as_bytes());
            }
            None => state.push(0),
        }
    }
    for path in [
        params.decoded().materials.normal_path.as_ref(),
        params.decoded().materials.roughness_path.as_ref(),
        params.decoded().materials.mask_path.as_ref(),
    ] {
        match path {
            Some(path) => {
                state.push(1);
                let content = std::fs::read(path).map_err(|error| {
                    PyRuntimeError::new_err(format!(
                        "failed to identity terrain material map {path:?}: {error}"
                    ))
                })?;
                let extension = std::path::Path::new(path)
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                append_content_digest(&mut state, extension.as_bytes(), &content);
            }
            None => state.push(0),
        }
    }
    scene
        .append_anamnesis_external_state(&mut state)
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))?;
    scene
        .append_anamnesis_external_state(&mut shadow_bytes)
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))?;
    super::anamnesis::TerrainCacheOptions::new(root, state, height_bytes, shadow_bytes)
        .map(Some)
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))
}

#[pymethods]
impl TerrainRenderer {
    #[new]
    pub fn new(session: &crate::core::session::Session) -> PyResult<Self> {
        let ctx = crate::core::gpu::try_ctx()?;
        let scene = TerrainScene::new(
            ctx.device.clone(),
            ctx.queue.clone(),
            session.adapter.clone(),
        )
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create TerrainRenderer: {:#}", e))
        })?;

        Ok(Self {
            scene,
            last_anamnesis_report: crate::core::anamnesis::CacheReport::default(),
            last_anamnesis_graph_submissions: 0,
        })
    }

    #[pyo3(signature = (lights))]
    #[allow(deprecated)]
    fn set_lights(&self, py: Python, lights: &PyAny) -> PyResult<()> {
        use pyo3::types::PyList;

        let lights_list = lights
            .downcast::<PyList>()
            .map_err(|_| PyRuntimeError::new_err("lights must be a list"))?;

        let mut native_lights = Vec::new();
        for (i, light_dict) in lights_list.iter().enumerate() {
            match crate::lighting::py_bindings::parse_light_dict(py, light_dict) {
                Ok(light) => native_lights.push(light),
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Failed to parse light {}: {}",
                        i, e
                    )));
                }
            }
        }

        {
            let mut override_guard = self.scene.light_override.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to lock light override: {}", e))
            })?;
            *override_guard = Some(native_lights.clone());
        }

        let mut light_buffer =
            self.scene.light_buffer.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to lock light buffer: {}", e))
            })?;

        light_buffer
            .update(&self.scene.device, &self.scene.queue, &native_lights)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update lights: {}", e)))?;

        Ok(())
    }

    #[pyo3(signature = () )]
    fn light_debug_info(&self) -> PyResult<String> {
        self.scene.light_debug_info()
    }

    #[pyo3(signature = (material_set, env_maps, params, heightmap, target=None, water_mask=None, time_seconds=0.0, certificate=None, cache=None))]
    pub fn render_terrain_pbr_pom<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        target: Option<&Bound<'_, PyAny>>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
        time_seconds: f32,
        certificate: Option<Bound<'py, PyAny>>,
        cache: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<crate::Frame>> {
        self.last_anamnesis_report = crate::core::anamnesis::CacheReport::default();
        self.last_anamnesis_graph_submissions = 0;
        if self
            .scene
            .offline_session_active()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query offline state: {e:#}")))?
        {
            return Err(PyRuntimeError::new_err(
                "An offline accumulation session is active; call end_offline_accumulation() before one-shot rendering.",
            ));
        }

        if target.is_some() {
            return Err(PyRuntimeError::new_err(
                "Custom render targets not yet supported. Use target=None for offscreen rendering.",
            ));
        }

        self.scene
            .validate_material_vt_request(params, material_set.materials().len() as u32)
            .map_err(|error| PyRuntimeError::new_err(format!("Rendering failed: {error:#}")))?;

        let certificate_enabled = certificate
            .as_ref()
            .is_some_and(|value| !value.is_none() && !matches!(value.extract::<bool>(), Ok(false)));
        let cache_options = terrain_cache_options(
            py,
            if certificate_enabled {
                None
            } else {
                cache.as_ref()
            },
            &self.scene,
            material_set,
            env_maps,
            params,
            &heightmap,
            water_mask.as_ref(),
            time_seconds,
        )?;
        let (frame, report, graph_submissions) = self
            .scene
            .render_internal(
                material_set,
                env_maps,
                params,
                heightmap,
                water_mask,
                time_seconds,
                cache_options.as_ref(),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering failed: {e:#}")))?;
        self.last_anamnesis_report = report;
        self.last_anamnesis_graph_submissions = graph_submissions;

        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;

        Py::new(py, frame)
    }

    #[pyo3(signature = (material_set, env_maps, params, heightmap, water_mask=None, time_seconds=0.0, certificate=None, cache=None))]
    pub fn render_with_aov<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
        time_seconds: f32,
        certificate: Option<Bound<'py, PyAny>>,
        cache: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Py<crate::Frame>, Py<crate::AovFrame>)> {
        self.last_anamnesis_report = crate::core::anamnesis::CacheReport::default();
        self.last_anamnesis_graph_submissions = 0;
        if cache.as_ref().is_some_and(|value| !value.is_none()) {
            return Err(PyRuntimeError::new_err(
                "native terrain ANAMNESIS does not yet serialize the multi-attachment AOV graph; use cache=None",
            ));
        }
        if self
            .scene
            .offline_session_active()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query offline state: {e:#}")))?
        {
            return Err(PyRuntimeError::new_err(
                "An offline accumulation session is active; call end_offline_accumulation() before one-shot rendering.",
            ));
        }

        self.scene
            .validate_material_vt_request(params, material_set.materials().len() as u32)
            .map_err(|error| {
                PyRuntimeError::new_err(format!("Rendering with AOV failed: {error:#}"))
            })?;

        let (frame, aov_frame) = self
            .scene
            .render_internal_with_aov(
                material_set,
                env_maps,
                params,
                heightmap,
                water_mask,
                time_seconds,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering with AOV failed: {:#}", e)))?;

        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok((Py::new(py, frame)?, Py::new(py, aov_frame)?))
    }

    pub fn info(&self) -> String {
        format!(
            "TerrainRenderer(backend=wgpu, device={:?})",
            self.scene.device.features()
        )
    }

    /// Report the most recent native one-shot ANAMNESIS lookup.
    #[getter]
    fn last_anamnesis_cache_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = &self.last_anamnesis_report;
        let dict = PyDict::new_bound(py);
        dict.set_item("hits", &report.hits)?;
        dict.set_item("misses", &report.misses)?;
        dict.set_item("bytes_read", report.bytes_read)?;
        dict.set_item("bytes_written", report.bytes_written)?;
        dict.set_item("wall_ms_saved", report.wall_ms_saved)?;
        dict.set_item("hit_rate", report.hit_rate())?;
        dict.set_item(
            "graph_command_submissions",
            self.last_anamnesis_graph_submissions,
        )?;
        Ok(dict.into())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = (batches))]
    pub fn set_scatter_batches(&mut self, batches: &PyAny) -> PyResult<()> {
        use pyo3::types::{PyDict, PyList};

        let batch_list = batches
            .downcast::<PyList>()
            .map_err(|_| PyRuntimeError::new_err("batches must be a list of dicts"))?;

        let mut native_batches = Vec::with_capacity(batch_list.len());
        for (batch_index, batch_any) in batch_list.iter().enumerate() {
            let batch_dict = batch_any.downcast::<PyDict>().map_err(|_| {
                PyRuntimeError::new_err(format!("scatter batch {batch_index} must be a dict"))
            })?;

            let name = batch_dict
                .get_item("name")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<String>())
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("batch {batch_index}: invalid 'name': {e}"))
                })?;

            let color = batch_dict
                .get_item("color")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<(f32, f32, f32, f32)>())
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("batch {batch_index}: invalid 'color': {e}"))
                })?
                .map(|value| [value.0, value.1, value.2, value.3])
                .unwrap_or([0.85, 0.85, 0.85, 1.0]);

            let max_draw_distance = batch_dict
                .get_item("max_draw_distance")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<f32>())
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index}: invalid 'max_draw_distance': {e}"
                    ))
                })?;
            let terrain_blend = if let Some(blend_any) = batch_dict
                .get_item("terrain_blend")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
            {
                let blend_dict = blend_any.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index}: 'terrain_blend' must be a dict"
                    ))
                })?;
                crate::terrain::scatter::TerrainScatterBlendConfig {
                    enabled: blend_dict
                        .get_item("enabled")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<bool>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_blend.enabled': {e}"
                            ))
                        })?
                        .unwrap_or(false),
                    bury_depth: blend_dict
                        .get_item("bury_depth")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<f32>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_blend.bury_depth': {e}"
                            ))
                        })?
                        .unwrap_or(0.75),
                    fade_distance: blend_dict
                        .get_item("fade_distance")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<f32>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_blend.fade_distance': {e}"
                            ))
                        })?
                        .unwrap_or(2.5),
                }
            } else {
                crate::terrain::scatter::TerrainScatterBlendConfig::default()
            };

            let terrain_contact = if let Some(contact_any) = batch_dict
                .get_item("terrain_contact")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
            {
                let contact_dict = contact_any.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index}: 'terrain_contact' must be a dict"
                    ))
                })?;
                crate::terrain::scatter::TerrainScatterContactConfig {
                    enabled: contact_dict
                        .get_item("enabled")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<bool>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_contact.enabled': {e}"
                            ))
                        })?
                        .unwrap_or(false),
                    distance: contact_dict
                        .get_item("distance")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<f32>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_contact.distance': {e}"
                            ))
                        })?
                        .unwrap_or(3.0),
                    strength: contact_dict
                        .get_item("strength")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<f32>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_contact.strength': {e}"
                            ))
                        })?
                        .unwrap_or(0.35),
                    vertical_weight: contact_dict
                        .get_item("vertical_weight")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .filter(|value| !value.is_none())
                        .map(|value| value.extract::<f32>())
                        .transpose()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'terrain_contact.vertical_weight': {e}"
                            ))
                        })?
                        .unwrap_or(0.65),
                }
            } else {
                crate::terrain::scatter::TerrainScatterContactConfig::default()
            };

            let transforms_any = batch_dict
                .get_item("transforms")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} missing required 'transforms'"
                    ))
                })?;
            let transforms_array: numpy::PyReadonlyArray2<'_, f32> =
                transforms_any.extract().map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index}: 'transforms' must be a float32 array: {e}"
                    ))
                })?;
            if transforms_array.ndim() != 2 || transforms_array.shape()[1] != 16 {
                return Err(PyRuntimeError::new_err(format!(
                    "batch {batch_index}: transforms must have shape (N, 16)"
                )));
            }
            let transforms = transforms_array
                .as_array()
                .rows()
                .into_iter()
                .map(|row| {
                    let mut out = [0.0f32; 16];
                    for (dst, src) in out.iter_mut().zip(row.iter()) {
                        *dst = *src;
                    }
                    out
                })
                .collect::<Vec<_>>();

            let levels_any = batch_dict
                .get_item("levels")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} missing required 'levels'"
                    ))
                })?;
            let levels_list = levels_any.downcast::<PyList>().map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "batch {batch_index}: 'levels' must be a list of dicts"
                ))
            })?;

            let mut levels = Vec::with_capacity(levels_list.len());
            for (level_index, level_any) in levels_list.iter().enumerate() {
                let level_dict = level_any.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} level {level_index} must be a dict"
                    ))
                })?;
                let mesh_any = level_dict
                    .get_item("mesh")
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index}: {e}"
                        ))
                    })?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index} missing required 'mesh'"
                        ))
                    })?;
                let mesh_dict = mesh_any.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} level {level_index}: 'mesh' must be a dict"
                    ))
                })?;
                let mesh = crate::geometry::mesh_from_python_dict(mesh_dict).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} level {level_index}: invalid mesh: {e}"
                    ))
                })?;
                let max_distance = level_dict
                    .get_item("max_distance")
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index}: {e}"
                        ))
                    })?
                    .filter(|value| !value.is_none())
                    .map(|value| value.extract::<f32>())
                    .transpose()
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index}: invalid 'max_distance': {e}"
                        ))
                    })?;
                levels
                    .push(crate::terrain::scatter::TerrainScatterLevelSpec { mesh, max_distance });
            }

            let wind = match batch_dict
                .get_item("wind")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|v| !v.is_none())
            {
                Some(wind_any) => {
                    let wind_dict = wind_any.downcast::<PyDict>().map_err(|_| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index}: 'wind' must be a dict"
                        ))
                    })?;

                    // Helper: extract a typed field from a dict, failing with a clear
                    // message on type mismatch (matching the fail-fast style of the
                    // rest of set_scatter_batches).  Missing keys use the default.
                    macro_rules! wind_field {
                        ($key:expr, $ty:ty, $default:expr) => {
                            match wind_dict
                                .get_item($key)
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!(
                                        "batch {batch_index} wind: {e}"
                                    ))
                                })?
                                .filter(|v| !v.is_none())
                            {
                                Some(v) => v.extract::<$ty>().map_err(|e| {
                                    PyRuntimeError::new_err(format!(
                                        "batch {batch_index} wind: invalid '{}': {e}",
                                        $key
                                    ))
                                })?,
                                None => $default,
                            }
                        };
                    }

                    crate::terrain::scatter::ScatterWindSettingsNative {
                        enabled: wind_field!("enabled", bool, false),
                        direction_deg: wind_field!("direction_deg", f32, 0.0),
                        speed: wind_field!("speed", f32, 1.0),
                        amplitude: wind_field!("amplitude", f32, 0.0),
                        rigidity: wind_field!("rigidity", f32, 0.5),
                        bend_start: wind_field!("bend_start", f32, 0.0),
                        bend_extent: wind_field!("bend_extent", f32, 1.0),
                        gust_strength: wind_field!("gust_strength", f32, 0.0),
                        gust_frequency: wind_field!("gust_frequency", f32, 0.3),
                        fade_start: wind_field!("fade_start", f32, 0.0),
                        fade_end: wind_field!("fade_end", f32, 0.0),
                    }
                }
                None => crate::terrain::scatter::ScatterWindSettingsNative::default(),
            };
            wind.validate()
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?;

            let hlod_config = batch_dict
                .get_item("hlod")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| -> PyResult<crate::terrain::scatter::HlodConfig> {
                    let d = value.downcast::<PyDict>().map_err(|_| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index}: 'hlod' must be a dict"
                        ))
                    })?;
                    let hlod_distance = d
                        .get_item("hlod_distance")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: 'hlod.hlod_distance' is required"
                            ))
                        })?;
                    let cluster_radius = d
                        .get_item("cluster_radius")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: 'hlod.cluster_radius' is required"
                            ))
                        })?;
                    let simplify_ratio = d
                        .get_item("simplify_ratio")
                        .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: 'hlod.simplify_ratio' is required"
                            ))
                        })?;
                    Ok(crate::terrain::scatter::HlodConfig {
                        hlod_distance: hlod_distance.extract().map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'hlod.hlod_distance': {e}"
                            ))
                        })?,
                        cluster_radius: cluster_radius.extract().map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'hlod.cluster_radius': {e}"
                            ))
                        })?,
                        simplify_ratio: simplify_ratio.extract().map_err(|e| {
                            PyRuntimeError::new_err(format!(
                                "batch {batch_index}: invalid 'hlod.simplify_ratio': {e}"
                            ))
                        })?,
                    })
                })
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("batch {batch_index}: invalid 'hlod': {e}"))
                })?;

            native_batches.push(super::scatter::TerrainScatterUploadBatch {
                name,
                color,
                max_draw_distance,
                terrain_blend,
                terrain_contact,
                transforms_rowmajor: transforms,
                levels,
                wind,
                hlod_config,
            });
        }

        self.scene
            .set_scatter_batches_native(native_batches)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set scatter batches: {e:#}")))
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = ())]
    pub fn clear_scatter_batches(&mut self) -> PyResult<()> {
        self.scene.clear_scatter_batches_native();
        Ok(())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = ())]
    pub fn get_scatter_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.scene.scatter_last_frame_stats();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("batch_count", stats.batch_count)?;
        dict.set_item("rendered_batches", stats.rendered_batches)?;
        dict.set_item("total_instances", stats.total_instances)?;
        dict.set_item("visible_instances", stats.visible_instances)?;
        dict.set_item("culled_instances", stats.culled_instances)?;
        dict.set_item("lod_instance_counts", stats.lod_instance_counts)?;
        dict.set_item("hlod_cluster_draws", stats.hlod_cluster_draws)?;
        dict.set_item("hlod_covered_instances", stats.hlod_covered_instances)?;
        dict.set_item("effective_draws", stats.effective_draws)?;
        Ok(dict.into())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = ())]
    pub fn get_scatter_memory_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = self.scene.scatter_memory_report();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("batch_count", report.batch_count)?;
        dict.set_item("level_count", report.level_count)?;
        dict.set_item("total_instances", report.total_instances)?;
        dict.set_item("vertex_buffer_bytes", report.vertex_buffer_bytes)?;
        dict.set_item("index_buffer_bytes", report.index_buffer_bytes)?;
        dict.set_item("instance_buffer_bytes", report.instance_buffer_bytes)?;
        dict.set_item("hlod_cluster_count", report.hlod_cluster_count)?;
        dict.set_item("hlod_buffer_bytes", report.hlod_buffer_bytes)?;
        dict.set_item("total_buffer_bytes", report.total_buffer_bytes())?;
        Ok(dict.into())
    }

    #[pyo3(signature = ())]
    pub fn get_probe_memory_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let probe_stride = std::mem::size_of::<crate::terrain::probes::GpuProbeData>() as u64;
        let probe_count = self
            .scene
            .probe_ssbo_bytes
            .checked_div(probe_stride)
            .unwrap_or(0);
        dict.set_item("probe_count", probe_count)?;
        dict.set_item("grid_uniform_bytes", self.scene.probe_grid_uniform_bytes)?;
        dict.set_item("probe_ssbo_bytes", self.scene.probe_ssbo_bytes)?;
        dict.set_item(
            "total_bytes",
            self.scene.probe_grid_uniform_bytes + self.scene.probe_ssbo_bytes,
        )?;
        Ok(dict.into())
    }

    #[pyo3(signature = ())]
    pub fn get_reflection_probe_memory_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("probe_count", self.scene.reflection_probe_count)?;
        dict.set_item("resolution", self.scene.reflection_probe_resolution)?;
        dict.set_item("mip_levels", self.scene.reflection_probe_mip_levels)?;
        dict.set_item(
            "grid_uniform_bytes",
            self.scene.reflection_probe_grid_uniform_bytes,
        )?;
        dict.set_item(
            "cubemap_texture_bytes",
            self.scene.reflection_probe_texture_bytes,
        )?;
        dict.set_item(
            "total_bytes",
            self.scene.reflection_probe_grid_uniform_bytes
                + self.scene.reflection_probe_texture_bytes,
        )?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainRenderer(features={:?})",
            self.scene.device.features()
        )
    }

    #[pyo3(
        text_signature = "(self, material_index, family, image_or_pyramid, virtual_size_px, fallback_color)"
    )]
    fn register_material_vt_source(
        &self,
        material_index: u32,
        family: String,
        image_or_pyramid: &Bound<'_, PyAny>,
        virtual_size_px: (u32, u32),
        fallback_color: Option<Vec<f32>>,
    ) -> PyResult<()> {
        use pyo3::prelude::*;

        // Extract image data from numpy array or bytes
        let data = if let Ok(arr) = image_or_pyramid.extract::<Vec<u8>>() {
            arr
        } else if let Ok(arr_any) = image_or_pyramid.getattr("tobytes") {
            arr_any.call0()?.extract::<Vec<u8>>()?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "image_or_pyramid must be bytes or numpy array",
            ));
        };

        // Parse fallback color
        let fallback = match fallback_color {
            Some(vec) if vec.len() >= 4 => [vec[0], vec[1], vec[2], vec[3]],
            _ => [0.5, 0.5, 0.5, 1.0],
        };

        let mut material_vt = self.scene.material_vt.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to lock material_vt: {}",
                e
            ))
        })?;

        material_vt
            .register_source(material_index, family, virtual_size_px, data, fallback)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    #[pyo3(text_signature = "(self)")]
    fn clear_material_vt_sources(&self) -> PyResult<()> {
        let mut material_vt = self.scene.material_vt.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to lock material_vt: {}",
                e
            ))
        })?;
        material_vt.clear_sources();
        Ok(())
    }

    #[pyo3(text_signature = "(self)")]
    fn get_material_vt_stats(&self) -> PyResult<std::collections::HashMap<String, f32>> {
        let material_vt = self.scene.material_vt.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to lock material_vt: {}",
                e
            ))
        })?;
        Ok(material_vt.get_stats())
    }

    /// Test hook for the mandatory live feedback-retention acceptance gate.
    /// The request is inserted into the active renderer VT runtime, held while
    /// feedback is forced not-ready, then converges through the normal upload
    /// path after the requested number of frames.
    #[pyo3(text_signature = "(self, not_ready_frames)")]
    fn force_vt_feedback_not_ready_for_test(&self, not_ready_frames: u32) -> PyResult<()> {
        let mut material_vt = self.scene.material_vt.lock().map_err(|error| {
            PyRuntimeError::new_err(format!("Failed to lock material_vt: {error}"))
        })?;
        material_vt
            .force_live_retention_probe(not_ready_frames)
            .map_err(PyRuntimeError::new_err)
    }

    /// TESSELLA win 6: the retained VT request set as
    /// `(family_slot, material_index, mip_level, tile_x, tile_y)` tuples.
    ///
    /// The retention gate compares this SET across a forced not-ready feedback
    /// window; `get_material_vt_stats()["retained_requests"]` only reports its
    /// cardinality, which cannot tell a preserved set from a silently swapped
    /// one of the same size.
    #[pyo3(text_signature = "(self)")]
    fn read_retained_vt_requests(&self) -> PyResult<Vec<(u32, u32, u32, u32, u32)>> {
        let material_vt = self.scene.material_vt.lock().map_err(|error| {
            PyRuntimeError::new_err(format!("Failed to lock material_vt: {error}"))
        })?;
        Ok(material_vt.retained_request_set())
    }

    /// Exact family-specific records emitted by the most recently staged GPU
    /// material-VT feedback pass. CPU prefetch and test-seeded requests are
    /// intentionally excluded.
    #[pyo3(text_signature = "(self)")]
    fn read_latest_vt_shader_feedback(&self) -> PyResult<Vec<(u32, u32, u32, u32, u32)>> {
        self.scene
            .drain_latest_material_vt_shader_feedback()
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "Failed to read material VT shader feedback: {error:#}"
                ))
            })
    }

    /// Return `(tile_id, triangle_id)` identities for visibility-buffer
    /// pixels. Background pixels return `None`.
    #[pyo3(text_signature = "(self, pixels)")]
    fn pick_visibility_pixels(&self, pixels: Vec<(u32, u32)>) -> PyResult<Vec<Option<(u32, u32)>>> {
        self.scene
            .pick_visibility_pixels(&pixels)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Independent CPU-BVH oracle for TESSELLA visibility-buffer acceptance.
    #[pyo3(text_signature = "(self, pixels)")]
    fn pick_visibility_pixels_cpu(
        &self,
        pixels: Vec<(u32, u32)>,
    ) -> PyResult<Vec<Option<(u32, u32)>>> {
        self.scene
            .pick_visibility_pixels_cpu(&pixels)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// VERITAS: resident-tile provenance for unresolved shader demand in the
    /// last rendered frame.
    ///
    /// Blocking drain of the VT feedback stream, resolved on the CPU to the
    /// resident mip each sampled texel actually landed on. Returns one dict
    /// per deduplicated tile: `{family, family_slot, source_id, tile_x,
    /// tile_y, mip_level, content_hash}` (hash hex-encoded). Empty when the
    /// terrain VT/feedback path is inactive or the latest frame had no
    /// unresolved desired pages.
    #[pyo3(text_signature = "(self)")]
    fn read_contributing_tiles(&self, py: Python<'_>) -> PyResult<PyObject> {
        use crate::core::provenance::{to_hex, FAMILY_NAMES};

        let tiles = self
            .scene
            .read_material_vt_contributing_tiles()
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to read contributing tiles: {e:#}"))
            })?;

        let out = pyo3::types::PyList::empty_bound(py);
        for tile in tiles {
            let dict = pyo3::types::PyDict::new_bound(py);
            let family = FAMILY_NAMES
                .get(tile.family_slot as usize)
                .copied()
                .unwrap_or("unknown");
            dict.set_item("family", family)?;
            dict.set_item("family_slot", tile.family_slot)?;
            dict.set_item("source_id", tile.source_id)?;
            dict.set_item("tile_x", tile.tile_x)?;
            dict.set_item("tile_y", tile.tile_y)?;
            dict.set_item("mip_level", tile.mip_level)?;
            dict.set_item("content_hash", to_hex(&tile.content_hash))?;
            out.append(dict)?;
        }
        Ok(out.into())
    }

    /// Resolve a retained raw VT shader-feedback snapshot against currently
    /// resident pages, preserving `read_contributing_tiles()` last-frame
    /// semantics. Feedback entries are `(family_slot, material_index,
    /// mip_level, tile_x, tile_y)`.
    #[pyo3(text_signature = "(self, feedback)")]
    fn resolve_captured_vt_feedback_provenance(
        &self,
        py: Python<'_>,
        feedback: Vec<(u32, u32, u32, u32, u32)>,
    ) -> PyResult<PyObject> {
        use crate::core::provenance::{to_hex, FAMILY_NAMES};

        let tiles = self
            .scene
            .resolve_captured_material_vt_feedback(&feedback)
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to resolve captured VT feedback provenance: {e:#}"
                ))
            })?;
        let out = pyo3::types::PyList::empty_bound(py);
        for tile in tiles {
            let dict = pyo3::types::PyDict::new_bound(py);
            let family = FAMILY_NAMES
                .get(tile.family_slot as usize)
                .copied()
                .unwrap_or("unknown");
            dict.set_item("family", family)?;
            dict.set_item("family_slot", tile.family_slot)?;
            dict.set_item("source_id", tile.source_id)?;
            dict.set_item("tile_x", tile.tile_x)?;
            dict.set_item("tile_y", tile.tile_y)?;
            dict.set_item("mip_level", tile.mip_level)?;
            dict.set_item("content_hash", to_hex(&tile.content_hash))?;
            out.append(dict)?;
        }
        Ok(out.into())
    }

    #[cfg(feature = "enable-renderer-config")]
    pub fn get_config(&self) -> PyResult<String> {
        let config = self
            .scene
            .config
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock config: {}", e)))?;
        serde_json::to_string_pretty(&*config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize config: {}", e)))
    }

    /// BOP-P2-02: enable runtime height-tile streaming for clipmap terrain.
    ///
    /// Builds a fixed-LOD `HeightMosaic` (`2^lod` tiles per axis at
    /// `tile_resolution` texels per tile), an `AsyncTileLoader` worker pool,
    /// and a `ClipmapStreamer` for camera-driven tile demand. When `dem` is
    /// given, tiles are bilinearly sliced from it on worker threads. Shipped
    /// render paths require an explicit source and never synthesize I/O. With
    /// `coarse_prefill=True` (default) every tile slot is filled from a
    /// low-resolution read so streaming frames show coarse terrain instead
    /// of holes while fine tiles are in flight.
    #[pyo3(signature = (terrain_extent_m, ring_count=4, ring_resolution=64, lod=2, tile_resolution=128, max_in_flight=16, pool_size=2, dem=None, coarse_prefill=true, max_resident_bytes=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn enable_height_streaming(
        &mut self,
        terrain_extent_m: f32,
        ring_count: u32,
        ring_resolution: u32,
        lod: u32,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        dem: Option<PyReadonlyArray2<f32>>,
        coarse_prefill: bool,
        max_resident_bytes: Option<u64>,
    ) -> PyResult<()> {
        use std::sync::Arc;

        if !(terrain_extent_m.is_finite() && terrain_extent_m > 0.0) {
            return Err(PyRuntimeError::new_err(
                "terrain_extent_m must be a positive finite value",
            ));
        }
        let lod = lod.min(6);
        let tile_resolution = tile_resolution.clamp(8, 1024);
        let reader: Arc<dyn crate::terrain::page_table::HeightReader> = match dem {
            Some(arr) => {
                let a = arr.as_array();
                let (h, w) = (a.shape()[0], a.shape()[1]);
                if h < 2 || w < 2 {
                    return Err(PyRuntimeError::new_err(
                        "dem must be at least 2x2 for bilinear tile slicing",
                    ));
                }
                Arc::new(super::streaming::DemSliceHeightReader::new(
                    a.iter().copied().collect(),
                    w,
                    h,
                ))
            }
            None => {
                return Err(PyRuntimeError::new_err(
                    "dem is required; synthetic height streaming is test-only",
                ))
            }
        };
        let state = super::streaming::HeightVtFamilyRuntime::new(
            self.scene.device.as_ref(),
            self.scene.queue.as_ref(),
            terrain_extent_m,
            ring_count,
            ring_resolution,
            lod,
            tile_resolution,
            max_in_flight,
            pool_size,
            reader,
            coarse_prefill,
            max_resident_bytes,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("enable_height_streaming failed: {:#}", e)))?;
        self.scene.height_streaming = Some(state);
        // Force clipmap mesh regeneration around the (new) streaming center.
        self.scene.geometry_provider = None;
        Ok(())
    }

    /// Enable the same height-mosaic path from a COG-backed
    /// `VirtualTextureStore`. The COG reader and the packed material store
    /// therefore share the page contract instead of maintaining a third
    /// render-time streamer.
    #[cfg(feature = "cog_streaming")]
    #[pyo3(signature = (dataset, terrain_extent_m, ring_count=4, ring_resolution=64, lod=2, tile_resolution=128, max_in_flight=16, pool_size=2, coarse_prefill=true, max_resident_bytes=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn enable_height_streaming_cog(
        &mut self,
        dataset: &crate::terrain::cog::py_bindings::PyCogDataset,
        terrain_extent_m: f32,
        ring_count: u32,
        ring_resolution: u32,
        lod: u32,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        coarse_prefill: bool,
        max_resident_bytes: Option<u64>,
    ) -> PyResult<()> {
        if !(terrain_extent_m.is_finite() && terrain_extent_m > 0.0) {
            return Err(PyRuntimeError::new_err(
                "terrain_extent_m must be a positive finite value",
            ));
        }
        let tile_resolution = tile_resolution.clamp(8, 1024);
        let store = std::sync::Arc::new(crate::terrain::vt::CogPageStore::from_reader(
            dataset.reader(),
            tile_resolution,
        ));
        let reader = std::sync::Arc::new(super::streaming::StoreHeightReader::new(store));
        let state = super::streaming::HeightVtFamilyRuntime::new(
            self.scene.device.as_ref(),
            self.scene.queue.as_ref(),
            terrain_extent_m,
            ring_count,
            ring_resolution,
            lod.min(6),
            tile_resolution,
            max_in_flight,
            pool_size,
            reader,
            coarse_prefill,
            max_resident_bytes,
        )
        .map_err(|error| {
            PyRuntimeError::new_err(format!("enable_height_streaming_cog failed: {error:#}"))
        })?;
        self.scene.height_streaming = Some(state);
        self.scene.geometry_provider = None;
        Ok(())
    }

    /// BOP-P2-02: drop the streaming state; renders fall back to the
    /// per-call overview heightmap.
    pub fn disable_height_streaming(&mut self) {
        self.scene.height_streaming = None;
        self.scene.geometry_provider = None;
    }

    /// BOP-P2-02: one streaming step — update the clipmap center from the
    /// camera position, request missing height tiles asynchronously, and
    /// drain completed tiles into the GPU mosaic. Returns a stats dict.
    #[pyo3(signature = (camera_pos, max_uploads=8))]
    pub fn stream_height_tiles(
        &mut self,
        py: Python<'_>,
        camera_pos: (f32, f32, f32),
        max_uploads: usize,
    ) -> PyResult<PyObject> {
        let queue = self.scene.queue.clone();
        let feedback_uvs = self
            .scene
            .material_vt
            .lock()
            .map_err(|_| PyRuntimeError::new_err("material VT mutex poisoned"))?
            .latest_feedback_uvs();
        let state = self.scene.height_streaming.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(
                "height streaming not enabled; call enable_height_streaming() first",
            )
        })?;
        let stats = state.stream_step(
            queue.as_ref(),
            glam::Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2),
            &feedback_uvs,
            max_uploads,
        );
        height_streaming_stats_to_py(py, &stats)
    }

    /// BOP-P2-02: current streaming stats without advancing the stream.
    pub fn height_streaming_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.scene.height_streaming.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "height streaming not enabled; call enable_height_streaming() first",
            )
        })?;
        height_streaming_stats_to_py(py, &state.stats())
    }
}

fn height_streaming_stats_to_py(
    py: Python<'_>,
    stats: &super::streaming::HeightStreamingStats,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("center", (stats.center.x, stats.center.y))?;
    dict.set_item("pending_ring_tiles", stats.pending_ring_tiles)?;
    dict.set_item("loaded_ring_tiles", stats.loaded_ring_tiles)?;
    dict.set_item("resident_fine_tiles", stats.resident_fine_tiles)?;
    dict.set_item("total_tiles", stats.total_tiles)?;
    dict.set_item("tiles_requested", stats.tiles_requested)?;
    dict.set_item("tiles_uploaded", stats.tiles_uploaded)?;
    dict.set_item("coarse_prefilled", stats.coarse_prefilled)?;
    dict.set_item("resident_height_bytes", stats.resident_height_bytes)?;
    dict.set_item("converged", stats.converged)?;
    dict.set_item("loader_pending", stats.loader_pending)?;
    dict.set_item("loader_completed", stats.loader_completed)?;
    Ok(dict.into())
}
