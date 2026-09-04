//! Narrow Python acceptance seam for PROMETHEUS's stochastic AETHER reference.

use super::super::super::*;

/// Run the independent GPU spectral atmosphere reference over a real DEM.
///
/// The returned `mean_xyz` is the unclipped per-pixel estimator and
/// `linear_rgb` is its untonemapped, non-negative public rendering form.
/// `variance` is the maximum per-pixel estimated variance of the sample-mean
/// luminance; the exact seed and SPP used by the dispatch are returned too.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    heightmap,
    width,
    height,
    cam,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    sun_azimuth_deg = 90.0,
    sun_elevation_deg = 10.0,
    sun_intensity = 20.0,
    turbidity = 2.0,
    ozone_du = 300.0,
    mie_g = 0.8,
    ground_albedo = 0.3,
    spp = 64u32,
    seed = 7u32,
    enabled = true,
    variance_threshold = 1e-3,
    certificate = None,
    cache = None,
))]
pub(crate) fn hybrid_render_aether_spectral_reference(
    py: Python<'_>,
    heightmap: numpy::PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    cam: &Bound<'_, PyDict>,
    spacing: (f32, f32),
    exaggeration: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
    ground_albedo: f32,
    spp: u32,
    seed: u32,
    enabled: bool,
    variance_threshold: f32,
    certificate: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use crate::path_tracing::hybrid_compute::{AetherSpectralReferenceDesc, HybridPathTracer};
    use numpy::PyArray1;

    // Pipeline/layout objects are immutable and belong to the process-lifetime
    // global GPU device. Keep one reference tracer alive across calls: on
    // Metal, destroying and recreating this large multi-entry pipeline graph
    // caused later command buffers to complete without executing the selected
    // entry point. Reusing the device-owned graph also matches the production
    // renderer lifecycle while every dispatch still binds fresh resources.
    static AETHER_REFERENCE_TRACER: once_cell::sync::OnceCell<HybridPathTracer> =
        once_cell::sync::OnceCell::new();

    // The acceptance reference is intentionally recomputed; accepting the
    // keyword keeps the CENSOR/ANAMNESIS public render contract uniform.
    let _ = cache;
    let capture =
        crate::core::certificate::begin_render_capture("hybrid_render_aether_spectral_reference");
    crate::core::gpu::try_ctx()?;
    let dem = heightmap.as_array();
    let (dem_height, dem_width) = (dem.shape()[0] as u32, dem.shape()[1] as u32);
    let vector = |key: &str, default: [f32; 3]| -> PyResult<[f32; 3]> {
        match cam.get_item(key)? {
            Some(value) => {
                let tuple: (f32, f32, f32) = value.extract()?;
                Ok([tuple.0, tuple.1, tuple.2])
            }
            None => Ok(default),
        }
    };
    let fov_y_deg = cam
        .get_item("fov_y")?
        .map(|value| value.extract::<f32>())
        .transpose()?
        .unwrap_or(20.0);
    let collect_timing = certificate
        .as_ref()
        .is_some_and(|value| !value.is_none() && !matches!(value.extract::<bool>(), Ok(false)));
    let desc = AetherSpectralReferenceDesc {
        heights: dem.iter().copied().collect(),
        dem_width,
        dem_height,
        spacing,
        exaggeration,
        cam_origin: vector("origin", [0.0, 1.0, 0.0])?,
        cam_look_at: vector("look_at", [1.0, 1.0, 0.0])?,
        cam_up: vector("up", [0.0, 1.0, 0.0])?,
        fov_y_deg,
        sun_azimuth_deg,
        sun_elevation_deg,
        sun_intensity,
        turbidity,
        ozone_du,
        mie_g,
        ground_albedo,
        width,
        height,
        seed,
        spp,
        enabled,
        variance_threshold,
        collect_timing,
    };
    let tracer = AETHER_REFERENCE_TRACER.get_or_try_init(HybridPathTracer::new)?;
    let output = tracer.render_aether_spectral_reference(&desc)?;
    let result = PyDict::new_bound(py);
    let mean_xyz = PyArray1::<f32>::from_vec_bound(py, output.mean_xyz).reshape([
        height as usize,
        width as usize,
        3,
    ])?;
    let rgb = PyArray1::<f32>::from_vec_bound(py, output.linear_rgb).reshape([
        height as usize,
        width as usize,
        3,
    ])?;
    result.set_item("mean_xyz", mean_xyz)?;
    result.set_item("linear_rgb", rgb)?;
    result.set_item("variance", output.variance)?;
    result.set_item("converged", output.converged)?;
    result.set_item("seed", output.seed)?;
    result.set_item("spp", output.spp)?;
    result.set_item("terrain_primary_hits", output.terrain_primary_hits)?;
    result.set_item("gpu_resource_bytes", output.gpu_resource_bytes)?;
    result.set_item("environment", "black")?;
    result.set_item("wavelength_count", crate::core::atmosphere::NUM_WAVELENGTHS)?;
    result.set_item("max_depth", 6u32)?;
    capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(result.into_py(py))
}
