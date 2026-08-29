//! Narrow PyO3 boundary for the canonical NEPHELE medium.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::media::{
    ratio_track, russian_roulette, Bounds3, DensityField, DensityMapping, DirectionalSun,
    EnvironmentDistribution, Grid3D, Homogeneous, MajorantGrid, Medium, PerlinWorley, Phase, Ray,
    Rgb, SampleIdentity, SpatialTransform, TrackingContext,
};

const PHYSICAL_SAMPLE_COUNT: usize = 1_000_000;

fn media_error(error: crate::media::MediaError) -> PyErr {
    PyValueError::new_err(format!("invalid participating medium: {error}"))
}

fn phase(kind: &str, g: f32) -> PyResult<Phase> {
    match kind {
        "isotropic" => Ok(Phase::Isotropic),
        "henyey_greenstein" | "hg" => Phase::henyey_greenstein(g).map_err(media_error),
        _ => Err(PyValueError::new_err(
            "phase must be 'isotropic' or 'henyey_greenstein'",
        )),
    }
}

fn mapping(scale: f32) -> DensityMapping {
    DensityMapping {
        physical_density_per_authored_unit: scale,
    }
}

fn sphere_exit_distance(point: [f32; 3], direction: [f32; 3], radius: f32) -> f32 {
    let projection = point
        .iter()
        .zip(direction)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let radius_remaining = radius * radius - point.iter().map(|value| value * value).sum::<f32>();
    -projection + (projection * projection + radius_remaining.max(0.0)).sqrt()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SphereOutcome {
    Transmitted,
    ScatteredOut,
    Absorbed,
}

fn sphere_transport(
    medium: &Medium,
    sample: u64,
    stream: u64,
) -> Result<SphereOutcome, crate::media::MediaError> {
    let radius = 5.0f32;
    let identity = SampleIdentity {
        frame: stream,
        pixel: 0,
        sample,
        bounce: 0,
    };
    let radial = identity.uniform(0).sqrt() * radius;
    let phi = std::f32::consts::TAU * identity.uniform(1);
    let mut point = [
        radial * phi.cos(),
        radial * phi.sin(),
        -(radius * radius - radial * radial).max(0.0).sqrt(),
    ];
    let mut direction = [0.0, 0.0, 1.0];
    let sigma_t = medium.sigma_t().components()[0];
    let density = medium.extinction_at([0.0; 3]).components()[0] / sigma_t;
    let rate = sigma_t * density;
    let albedo = medium.sigma_s().components()[0] / sigma_t;
    let mut collision_count = 0u32;
    loop {
        let bounce_identity = SampleIdentity {
            bounce: collision_count,
            ..identity
        };
        let free_flight = -bounce_identity.uniform(2).ln() / rate;
        let exit = sphere_exit_distance(point, direction, radius);
        if free_flight >= exit {
            return Ok(if collision_count == 0 {
                SphereOutcome::Transmitted
            } else {
                SphereOutcome::ScatteredOut
            });
        }
        point = std::array::from_fn(|axis| point[axis] + direction[axis] * free_flight);
        if bounce_identity.uniform(3) >= albedo {
            return Ok(SphereOutcome::Absorbed);
        }
        direction = medium
            .phase()
            .sample(
                direction,
                [bounce_identity.uniform(4), bounce_identity.uniform(5)],
            )?
            .direction;
        collision_count = collision_count.wrapping_add(1);
    }
}

#[pyclass(module = "forge3d.media", name = "Medium", frozen)]
#[derive(Clone)]
pub struct PyMedium {
    medium: Medium,
    version: u64,
}

impl PyMedium {
    pub(crate) fn medium(&self) -> &Medium {
        &self.medium
    }

    pub(crate) fn version_value(&self) -> u64 {
        self.version
    }
}

#[pymethods]
impl PyMedium {
    #[staticmethod]
    #[pyo3(signature = (sigma_a, sigma_s, density=1.0, *, phase="isotropic", g=0.0, density_scale=1.0, version=0))]
    fn homogeneous(
        sigma_a: [f32; 3],
        sigma_s: [f32; 3],
        density: f32,
        phase: &str,
        g: f32,
        density_scale: f32,
        version: u64,
    ) -> PyResult<Self> {
        let density = DensityField::Homogeneous(Homogeneous {
            authored_density: density,
            mapping: mapping(density_scale),
        });
        Ok(Self {
            medium: Medium::new(sigma_a, sigma_s, self::phase(phase, g)?, density)
                .map_err(media_error)?,
            version,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (sigma_a, sigma_s, density, bounds, *, phase="isotropic", g=0.0, density_scale=1.0, version=0))]
    fn grid3d(
        sigma_a: [f32; 3],
        sigma_s: [f32; 3],
        density: PyReadonlyArray3<'_, f32>,
        bounds: ([f32; 3], [f32; 3]),
        phase: &str,
        g: f32,
        density_scale: f32,
        version: u64,
    ) -> PyResult<Self> {
        let shape = density.shape();
        let dimensions = [shape[2] as u32, shape[1] as u32, shape[0] as u32];
        let values = density
            .as_slice()
            .map_err(|_| PyValueError::new_err("density must be a contiguous float32 array"))?
            .to_vec();
        let field = Grid3D::new(
            SpatialTransform {
                bounds: Bounds3 {
                    min: bounds.0,
                    max: bounds.1,
                },
            },
            dimensions,
            values,
            mapping(density_scale),
        )
        .map(DensityField::Grid3D)
        .map_err(media_error)?;
        Ok(Self {
            medium: Medium::new(sigma_a, sigma_s, self::phase(phase, g)?, field)
                .map_err(media_error)?,
            version,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (sigma_a, sigma_s, bounds, *, frequency, octaves, seed, worley_weight, phase="isotropic", g=0.0, density_scale=1.0, version=0))]
    #[allow(clippy::too_many_arguments)]
    fn perlin_worley(
        sigma_a: [f32; 3],
        sigma_s: [f32; 3],
        bounds: ([f32; 3], [f32; 3]),
        frequency: f32,
        octaves: u32,
        seed: u64,
        worley_weight: f32,
        phase: &str,
        g: f32,
        density_scale: f32,
        version: u64,
    ) -> PyResult<Self> {
        let field = DensityField::PerlinWorley(PerlinWorley {
            transform: SpatialTransform {
                bounds: Bounds3 {
                    min: bounds.0,
                    max: bounds.1,
                },
            },
            frequency,
            octaves,
            seed,
            worley_weight,
            mapping: mapping(density_scale),
        });
        Ok(Self {
            medium: Medium::new(sigma_a, sigma_s, self::phase(phase, g)?, field)
                .map_err(media_error)?,
            version,
        })
    }

    #[getter]
    fn sigma_a(&self) -> [f32; 3] {
        self.medium.sigma_a().components()
    }

    #[getter]
    fn sigma_s(&self) -> [f32; 3] {
        self.medium.sigma_s().components()
    }

    #[getter]
    fn sigma_t(&self) -> [f32; 3] {
        self.medium.sigma_t().components()
    }

    #[getter(version)]
    fn py_version(&self) -> u64 {
        self.version
    }

    #[getter]
    fn identity(&self) -> String {
        let identity = self.medium.identity(self.version);
        identity
            .digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value = serde_json::to_string(&self.medium)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let result = py.import_bound("json")?.call_method1("loads", (value,))?;
        result.set_item("version", self.version)?;
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Medium(identity='{}', version={})",
            self.identity(),
            self.version
        )
    }
}

/// Integrated slow terrain/media reference using the live HybridPathTracer
/// terrain adapter for every acceptance pixel and sample.
#[pyfunction]
#[pyo3(name = "_render_volumetric_reference", signature = (medium, heightmap, width, height, camera, *, spacing=(1.0, 1.0), exaggeration=1.0, albedo=(0.6, 0.6, 0.6), sun_azimuth_deg=315.0, sun_elevation_deg=45.0, sun_intensity=2.5, sun_color=(1.0, 0.97, 0.92), environment_intensity=0.35, exposure=1.0, samples_per_pixel=1, homogeneous_medium_reach, clip=(0.1, 6000.0), seed=7, certificate=None, cache=None, full_viewport=None, crop=None))]
#[allow(clippy::too_many_arguments)]
fn render_volumetric_reference(
    py: Python<'_>,
    medium: &PyMedium,
    heightmap: PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    camera: &Bound<'_, PyDict>,
    spacing: (f32, f32),
    exaggeration: f32,
    albedo: (f32, f32, f32),
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    sun_color: (f32, f32, f32),
    environment_intensity: f32,
    exposure: f32,
    samples_per_pixel: u32,
    homogeneous_medium_reach: f32,
    clip: (f32, f32),
    seed: u32,
    certificate: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
    full_viewport: Option<Vec<u32>>,
    crop: Option<Vec<u32>>,
) -> PyResult<PyObject> {
    let _ = cache;
    let capture = crate::core::certificate::begin_render_capture("render_volumetric_reference");
    let majorant =
        MajorantGrid::construct(medium.medium(), medium.version_value()).map_err(media_error)?;
    let context = TrackingContext::new(medium.medium().clone(), majorant).map_err(media_error)?;
    let orbit = camera
        .get_item("terrain_camera")?
        .ok_or_else(|| PyValueError::new_err("camera.terrain_camera is required"))?;
    let orbit = orbit.downcast::<PyDict>()?;
    let get_orbit = |key: &str| -> PyResult<Bound<'_, PyAny>> {
        orbit.get_item(key)?.ok_or_else(|| {
            PyValueError::new_err(format!("camera.terrain_camera.{key} is required"))
        })
    };
    let target_tuple = get_orbit("target")?.extract::<(f32, f32, f32)>()?;
    let target = glam::Vec3::new(target_tuple.0, target_tuple.1, target_tuple.2);
    let radius = get_orbit("radius")?.extract::<f32>()?;
    let phi_deg = get_orbit("phi_deg")?.extract::<f32>()?;
    let theta_deg = get_orbit("theta_deg")?.extract::<f32>()?;
    let mode = get_orbit("mode")?.extract::<String>()?;
    if !crate::terrain::is_yup_camera_mode(&mode) {
        return Err(PyValueError::new_err(
            "reference terrain camera requires the authoritative mesh:yup frame",
        ));
    }
    let fov_y_deg = camera
        .get_item("fov_y")?
        .ok_or_else(|| PyValueError::new_err("camera.fov_y is required"))?
        .extract::<f32>()?;
    let dem = heightmap.as_array();
    let full_viewport = match full_viewport {
        Some(values) if values.len() == 2 => [values[0], values[1]],
        Some(_) => {
            return Err(PyValueError::new_err(
                "full_viewport must contain exactly (width, height)",
            ))
        }
        None => [width, height],
    };
    let crop_origin = match crop {
        Some(values) if values.len() == 4 => {
            if values[2] != width || values[3] != height {
                return Err(PyValueError::new_err(
                    "crop width and height must match the requested output width and height",
                ));
            }
            [values[0], values[1]]
        }
        Some(_) => {
            return Err(PyValueError::new_err(
                "crop must contain exactly (x, y, width, height)",
            ))
        }
        None => [0, 0],
    };
    let (eye, camera_view, _) = crate::terrain::build_orbit_view_proj(
        target,
        radius,
        phi_deg,
        theta_deg,
        fov_y_deg,
        full_viewport[0] as f32 / full_viewport[1] as f32,
        clip.0,
        clip.1,
    );
    let forward = (target - eye).normalize();
    let right = glam::Vec3::new(
        camera_view.x_axis.x,
        camera_view.y_axis.x,
        camera_view.z_axis.x,
    );
    let camera_up = glam::Vec3::new(
        camera_view.x_axis.y,
        camera_view.y_axis.y,
        camera_view.z_axis.y,
    );
    let desc = crate::path_tracing::hybrid_compute::TerrainReferenceDesc {
        heights: dem.iter().copied().collect(),
        dem_width: dem.shape()[1] as u32,
        dem_height: dem.shape()[0] as u32,
        spacing,
        exaggeration,
        albedo: [albedo.0, albedo.1, albedo.2],
        cam_origin: eye.to_array(),
        cam_look_at: target.to_array(),
        cam_up: camera_up.to_array(),
        fov_y_deg,
        exposure,
        sun_azimuth_deg,
        sun_elevation_deg,
        sun_intensity,
        sun_color: [sun_color.0, sun_color.1, sun_color.2],
        observer_geodetic_deg: [0.0; 2],
        earth_model: crate::geo::refraction::EarthModel::Flat,
        refraction_model: crate::geo::refraction::RefractionModel::None,
        env_map: None,
        env_intensity: environment_intensity,
        atmosphere: None,
        mesh: None,
        width,
        height,
        seed,
        spp: samples_per_pixel,
        max_frames: 1,
        min_frames: 1,
        variance_threshold: 1.0,
    };
    let environment = EnvironmentDistribution::new(
        1,
        1,
        vec![Rgb::new([environment_intensity; 3], "environment intensity").map_err(media_error)?],
    )
    .map_err(media_error)?;
    let azimuth = sun_azimuth_deg.to_radians();
    let elevation = sun_elevation_deg.to_radians();
    let sun = DirectionalSun::new(
        [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        ],
        [
            sun_color.0 * sun_intensity,
            sun_color.1 * sun_intensity,
            sun_color.2 * sun_intensity,
        ],
    )
    .map_err(media_error)?;
    let tracer = crate::path_tracing::hybrid_compute::HybridPathTracer::new()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let output = tracer
        .render_terrain_media_reference(
            &desc,
            &context,
            homogeneous_medium_reach,
            &environment,
            sun,
            u64::from(samples_per_pixel),
            clip,
            crate::media::ReferenceTransportConfig {
                roulette_start_bounce: 3,
                roulette_minimum_probability: 0.05,
            },
            full_viewport,
            crop_origin,
            [forward.to_array(), right.to_array(), camera_up.to_array()],
        )
        .map_err(media_error)?;
    let dict = PyDict::new_bound(py);
    let camera_contract = PyDict::new_bound(py);
    camera_contract.set_item("origin", eye.to_array())?;
    camera_contract.set_item("look_at", target.to_array())?;
    camera_contract.set_item("up", camera_up.to_array())?;
    camera_contract.set_item("right", right.to_array())?;
    camera_contract.set_item("forward", forward.to_array())?;
    camera_contract.set_item("fov_y", fov_y_deg)?;
    dict.set_item("camera_contract", camera_contract)?;
    let shape = [height as usize, width as usize, 3];
    dict.set_item(
        "beauty",
        PyArray1::from_vec_bound(py, output.beauty).reshape(shape)?,
    )?;
    dict.set_item(
        "transmittance",
        PyArray1::from_vec_bound(py, output.transmittance).reshape(shape)?,
    )?;
    dict.set_item(
        "in_scatter",
        PyArray1::from_vec_bound(py, output.in_scatter).reshape(shape)?,
    )?;
    dict.set_item(
        "cloud_shadow",
        PyArray1::from_vec_bound(py, output.cloud_shadow).reshape(shape)?,
    )?;
    dict.set_item(
        "optical_depth",
        PyArray1::from_vec_bound(py, output.optical_depth).reshape(shape)?,
    )?;
    dict.set_item(
        "terrain_slice",
        PyArray1::from_vec_bound(py, output.terrain_slice)
            .reshape([height as usize, width as usize])?,
    )?;
    dict.set_item(
        "terrain_hit",
        PyArray1::from_vec_bound(py, output.terrain_hit)
            .reshape([height as usize, width as usize])?,
    )?;
    dict.set_item(
        "media_lighting_visibility",
        PyArray1::from_vec_bound(py, output.media_lighting_visibility)
            .reshape([height as usize, width as usize])?,
    )?;
    let diagnostics = PyDict::new_bound(py);
    let majorant_proof = serde_json::to_string(&output.majorant_proof)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    diagnostics.set_item(
        "majorant_proof",
        py.import_bound("json")?
            .call_method1("loads", (majorant_proof,))?,
    )?;
    diagnostics.set_item("majorant_valid", true)?;
    diagnostics.set_item("sample_count", output.sample_count)?;
    diagnostics.set_item("step_count", output.step_count)?;
    diagnostics.set_item("temporal_history_decision", "not_applicable")?;
    diagnostics.set_item(
        "temporal_history_reason",
        "independent reference samples do not reuse temporal history",
    )?;
    diagnostics.set_item("host_visible_bytes", output.host_visible_bytes)?;
    diagnostics.set_item("froxel_device_local_bytes", 0u64)?;
    diagnostics.set_item("density_device_local_bytes", 0u64)?;
    diagnostics.set_item("majorant_device_local_bytes", 0u64)?;
    diagnostics.set_item("staging_readback_bytes", 0u64)?;
    diagnostics.set_item("adapter", output.adapter)?;
    diagnostics.set_item("backend", output.backend)?;
    diagnostics.set_item("driver", output.driver)?;
    diagnostics.set_item("source_revision", env!("FORGE3D_GIT_SHA_FULL"))?;
    diagnostics.set_item("executed_multi_scatter", output.executed_multi_scatter)?;
    diagnostics.set_item("single_scatter_luminance", py.None())?;
    diagnostics.set_item("multiple_scatter_luminance", py.None())?;
    diagnostics.set_item("energy_accounting_residual", py.None())?;
    dict.set_item("diagnostics", diagnostics)?;
    capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(dict.into())
}

/// Acceptance-only execution of the canonical tracking and roulette kernels.
/// Raw samples are returned so the external verifier can recompute every
/// accumulator instead of trusting native summary claims.
#[pyfunction]
#[pyo3(name = "_nephele_physical_samples", signature = (homogeneous, heterogeneous, homogeneous_distance, sample_count))]
fn nephele_physical_samples(
    py: Python<'_>,
    homogeneous: &PyMedium,
    heterogeneous: &PyMedium,
    homogeneous_distance: f32,
    sample_count: usize,
) -> PyResult<PyObject> {
    if sample_count != PHYSICAL_SAMPLE_COUNT {
        return Err(PyValueError::new_err(
            "NEPHELE physical sampling requires exactly 1,000,000 samples",
        ));
    }
    if !homogeneous_distance.is_finite() || homogeneous_distance <= 0.0 {
        return Err(PyValueError::new_err(
            "homogeneous slab distance must be finite and positive",
        ));
    }
    let homogeneous_majorant =
        MajorantGrid::construct(homogeneous.medium(), homogeneous.version_value())
            .map_err(media_error)?;
    let homogeneous_context =
        TrackingContext::new(homogeneous.medium().clone(), homogeneous_majorant)
            .map_err(media_error)?;
    let heterogeneous_majorant =
        MajorantGrid::construct(heterogeneous.medium(), heterogeneous.version_value())
            .map_err(media_error)?;
    let heterogeneous_context =
        TrackingContext::new(heterogeneous.medium().clone(), heterogeneous_majorant)
            .map_err(media_error)?;
    let bounds = match heterogeneous.medium().density() {
        DensityField::Grid3D(field) => field.transform().bounds,
        _ => {
            return Err(PyValueError::new_err(
                "heterogeneous physical sampling requires a Grid3D medium",
            ))
        }
    };
    let distance = bounds.max[2] - bounds.min[2];
    let sigma_t = homogeneous.medium().sigma_t().components()[0];
    if sigma_t <= 0.0
        || homogeneous
            .medium()
            .sigma_t()
            .components()
            .iter()
            .any(|value| *value != sigma_t)
        || homogeneous
            .medium()
            .sigma_s()
            .components()
            .iter()
            .any(|value| *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "physical slab requires a positive spectrally homogeneous extinction",
        ));
    }
    let albedo = homogeneous.medium().sigma_s().components()[0] / sigma_t;
    let mut homogeneous_values = Vec::with_capacity(sample_count);
    let mut heterogeneous_values = Vec::with_capacity(sample_count);
    let mut rr_on = Vec::with_capacity(sample_count);
    let mut rr_off = Vec::with_capacity(sample_count);
    let mut transmitted = Vec::with_capacity(sample_count);
    let mut scattered_out = Vec::with_capacity(sample_count);
    let mut absorbed = Vec::with_capacity(sample_count);
    let ray = Ray {
        origin: [0.0; 3],
        direction: [1.0, 0.0, 0.0],
    };
    for sample in 0..sample_count as u64 {
        let homogeneous_identity = SampleIdentity {
            frame: 0x4e45_5048_1001,
            pixel: 0,
            sample,
            bounce: 0,
        };
        homogeneous_values.push(
            ratio_track(
                &homogeneous_context,
                ray,
                homogeneous_distance,
                homogeneous_identity,
            )
            .map_err(media_error)?
            .0
            .components()[0] as f64,
        );
        let coordinate_identity = SampleIdentity {
            frame: 0x4e45_5048_1101,
            pixel: 0,
            sample,
            bounce: 0,
        };
        let heterogeneous_ray = Ray {
            origin: [
                bounds.min[0] + (bounds.max[0] - bounds.min[0]) * coordinate_identity.uniform(0),
                bounds.min[1] + (bounds.max[1] - bounds.min[1]) * coordinate_identity.uniform(1),
                bounds.min[2],
            ],
            direction: [0.0, 0.0, 1.0],
        };
        heterogeneous_values.push(
            ratio_track(
                &heterogeneous_context,
                heterogeneous_ray,
                distance,
                SampleIdentity {
                    frame: 0x4e45_5048_1102,
                    ..coordinate_identity
                },
            )
            .map_err(media_error)?
            .0
            .components()[0] as f64,
        );

        let collision = crate::media::delta_track(
            &homogeneous_context,
            ray,
            homogeneous_distance,
            0,
            SampleIdentity {
                frame: 0x4e45_5048_1201,
                ..homogeneous_identity
            },
        )
        .map_err(media_error)?
        .is_some();
        rr_off.push(if collision { f64::from(albedo) } else { 0.0 });
        let mut roulette_contribution = 0.0f64;
        if collision {
            let throughput =
                Rgb::new([albedo; 3], "single-scatter slab albedo").map_err(media_error)?;
            for trial in 0..4u64 {
                if let Some(scale) = russian_roulette(
                    throughput,
                    homogeneous_identity.uniform(0x2000 + trial),
                    0.0,
                )
                .map_err(media_error)?
                {
                    roulette_contribution += f64::from(albedo * scale) * 0.25;
                }
            }
        }
        rr_on.push(roulette_contribution);

        transmitted.push(
            (sphere_transport(homogeneous.medium(), sample, 0x4e45_5048_2001)
                .map_err(media_error)?
                == SphereOutcome::Transmitted) as u8 as f64,
        );
        scattered_out.push(
            (sphere_transport(homogeneous.medium(), sample, 0x4e45_5048_2002)
                .map_err(media_error)?
                == SphereOutcome::ScatteredOut) as u8 as f64,
        );
        absorbed.push(
            (sphere_transport(homogeneous.medium(), sample, 0x4e45_5048_2003)
                .map_err(media_error)?
                == SphereOutcome::Absorbed) as u8 as f64,
        );
    }
    let result = PyDict::new_bound(py);
    result.set_item(
        "homogeneous_ratio",
        PyArray1::from_vec_bound(py, homogeneous_values),
    )?;
    result.set_item(
        "heterogeneous_ratio",
        PyArray1::from_vec_bound(py, heterogeneous_values),
    )?;
    result.set_item("rr_on", PyArray1::from_vec_bound(py, rr_on))?;
    result.set_item("rr_off", PyArray1::from_vec_bound(py, rr_off))?;
    result.set_item("transmitted", PyArray1::from_vec_bound(py, transmitted))?;
    result.set_item("scattered_out", PyArray1::from_vec_bound(py, scattered_out))?;
    result.set_item("absorbed", PyArray1::from_vec_bound(py, absorbed))?;
    result.set_item("source_revision", env!("FORGE3D_GIT_SHA_FULL"))?;
    result.set_item(
        "implementation",
        "canonical-ratio-delta-roulette-and-analog-sphere-v1",
    )?;
    Ok(result.into())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMedium>()?;
    module.add_function(wrap_pyfunction!(render_volumetric_reference, module)?)?;
    module.add_function(wrap_pyfunction!(nephele_physical_samples, module)?)?;
    Ok(())
}
