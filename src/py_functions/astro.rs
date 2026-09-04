use super::super::*;
use crate::{
    astro::{self, time::UtcDateTime, Body, Observer},
    geo::units::{Angle, Degree},
};
use glam::DMat3;

fn inputs(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_m: f64,
) -> PyResult<(UtcDateTime, Observer)> {
    let utc = UtcDateTime::new(year, month, day, hour, minute, second)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let observer = Observer::new(
        Angle::new(latitude_deg),
        Angle::new(longitude_deg),
        height_m,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((utc, observer))
}

#[pyfunction]
#[pyo3(signature = (
    body, year, month, day, hour, minute, second, latitude_deg, longitude_deg,
    height_m=0.0, refraction=false
))]
pub(crate) fn astro_body_position(
    body: &str,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_m: f64,
    refraction: bool,
) -> PyResult<(f64, f64, f64)> {
    let (utc, observer) = inputs(
        year,
        month,
        day,
        hour,
        minute,
        second,
        latitude_deg,
        longitude_deg,
        height_m,
    )?;
    let position = astro::body_position(
        Body::parse(body).map_err(|error| PyValueError::new_err(error.to_string()))?,
        utc,
        observer,
        refraction,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        position.azimuth.value(),
        position.altitude.value(),
        position.distance_au,
    ))
}

#[pyfunction]
pub(crate) fn astro_moon_phase(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> PyResult<(f64, f64, f64)> {
    let utc = UtcDateTime::new(year, month, day, hour, minute, second)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let phase = astro::moon_phase(utc).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        phase.illuminated_fraction,
        phase.phase_angle.value(),
        phase.apparent_semidiameter_arcsec,
    ))
}

/// ΔT = TT − UT1 in seconds from the committed piecewise-linear fit.
///
/// Exposed because SIDERA declares a residual for this quantity and the gate
/// that proves the declaration lives in `tests/test_astro_ephemeris.py`, where
/// it is compared against JPL Horizons' own time-scale columns.
#[pyfunction]
pub(crate) fn astro_delta_t_seconds(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> PyResult<f64> {
    let utc = UtcDateTime::new(year, month, day, hour, minute, second)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    astro::time::delta_t_seconds(utc).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Greenwich mean and apparent sidereal time in degrees, IAU 2006 + the
/// declared nutation terms. See `src/astro/frames.rs`.
#[pyfunction]
pub(crate) fn astro_sidereal_time(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> PyResult<(f64, f64)> {
    let utc = UtcDateTime::new(year, month, day, hour, minute, second)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let jd_ut1 = astro::time::julian_day_ut1(utc)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let jd_tt = astro::time::julian_day_tt(utc)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        astro::frames::gmst(jd_ut1, jd_tt).to_degrees(),
        astro::frames::gast(jd_ut1, jd_tt).to_degrees(),
    ))
}

/// Atmospheric refraction in arcminutes for a *true* altitude, Sæmundsson
/// (1986) at the declared standard atmosphere. See `src/astro/frames.rs`.
#[pyfunction]
pub(crate) fn astro_refraction_arcminutes(true_altitude_deg: f64) -> PyResult<f64> {
    if !true_altitude_deg.is_finite() {
        return Err(PyValueError::new_err("true_altitude_deg must be finite"));
    }
    Ok(
        (astro::frames::refract_altitude(Angle::<Degree>::new(true_altitude_deg)).value()
            - true_altitude_deg)
            * 60.0,
    )
}

#[pyfunction]
#[pyo3(signature = (
    year, month, day, hour, minute, second, latitude_deg, longitude_deg, height_m=0.0
))]
pub(crate) fn sky_set_observation(
    py: Python<'_>,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_m: f64,
) -> PyResult<Py<PyDict>> {
    let (utc, observer) = inputs(
        year,
        month,
        day,
        hour,
        minute,
        second,
        latitude_deg,
        longitude_deg,
        height_m,
    )?;
    let observation = astro::observation::set_observation(utc, observer)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = PyDict::new_bound(py);
    result.set_item("revision", observation.revision)?;
    result.set_item("star_count", observation.stars.len())?;
    for (body, position) in observation.bodies {
        result.set_item(
            body.name(),
            (
                position.azimuth.value(),
                position.altitude.value(),
                position.distance_au,
            ),
        )?;
    }
    result.set_item(
        "moon_phase",
        (
            observation.moon_phase.illuminated_fraction,
            observation.moon_phase.phase_angle.value(),
            observation.moon_phase.apparent_semidiameter_arcsec,
        ),
    )?;
    Ok(result.unbind())
}

/// CPU-only validation metrics backing DoD gates 4 and 5.
///
/// Nothing here compares SIDERA against a stored copy of its own output. The
/// sidereal-time gate runs the ERA-based IAU 2006 kernel against an
/// independently coded IAU 1982 (Aoki et al.) polynomial across the whole
/// window, and the refraction gate runs the Sæmundsson fit — inverted to the
/// apparent-altitude framing the DoD asks for — against Bennett's independent
/// fit at 5° apparent altitude.
#[pyfunction]
pub(crate) fn astro_validation_metrics(py: Python<'_>) -> PyResult<Py<PyDict>> {
    use astro::time::{julian_day_tt, julian_day_ut1};

    // Sweep the closed window at a stride that is coprime with both the
    // sidereal and the tropical year so the sample set does not alias onto a
    // fixed phase of either.
    const GMST_SWEEP_STRIDE_DAYS: f64 = 11.0;
    let window_start = julian_day_ut1(
        UtcDateTime::new(2000, 1, 1, 0, 0, 0.0)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let window_end = julian_day_ut1(
        UtcDateTime::new(2050, 12, 31, 23, 0, 0.0)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let mut gmst_max_seconds = 0.0_f64;
    let mut gmst_samples = 0u32;
    let mut jd_ut1 = window_start;
    while jd_ut1 <= window_end {
        // TT-UT1 is ~69 s here; the IAU 2006 GMST precession argument is a
        // century-scale polynomial, so using UT1 for it costs far below a
        // microsecond and keeps the sweep independent of the ΔT table.
        let modern = astro::frames::gmst(jd_ut1, jd_ut1) * 86_400.0 / std::f64::consts::TAU;
        let classical = astro::frames::gmst_iau1982_seconds(jd_ut1);
        let mut delta = modern - classical;
        delta -= (delta / 86_400.0).round() * 86_400.0;
        gmst_max_seconds = gmst_max_seconds.max(delta.abs());
        gmst_samples += 1;
        jd_ut1 += GMST_SWEEP_STRIDE_DAYS;
    }

    // DoD gate 4 is stated at 5° *apparent* altitude, so invert SIDERA's
    // true->apparent fit before comparing with Bennett's apparent->refraction
    // fit at the same apparent altitude.
    const REFRACTION_GATE_APPARENT_DEG: f64 = 5.0;
    let gate_apparent = Angle::<Degree>::new(REFRACTION_GATE_APPARENT_DEG);
    let true_altitude = astro::frames::unrefract_altitude(gate_apparent).ok_or_else(|| {
        PyRuntimeError::new_err("5 deg apparent altitude is outside the refraction fit's image")
    })?;
    let sidera_refraction_arcminutes =
        (REFRACTION_GATE_APPARENT_DEG - true_altitude.value()) * 60.0;
    let bennett_refraction_arcminutes = astro::frames::bennett_refraction_arcminutes(gate_apparent);
    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let jd_tt = julian_day_tt(utc).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let observer = Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    // DoD gate 5 is about *star* positions, so measure it on the whole rendered
    // catalog through the real reduction chain, not on a synthetic axis vector.
    let (precession_min, precession_median, precession_max) =
        astro::catalog::precession_ablation_arcminutes(utc, observer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let sidereal = astro::frames::gast(
        julian_day_ut1(utc).map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
        jd_tt,
    );
    let lunar = astro::moon::geocentric_ecliptic(jd_tt)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let (dpsi, deps, obliquity) = astro::frames::nutation(jd_tt);
    let geocentric = DMat3::from_rotation_x(obliquity + deps)
        * DMat3::from_rotation_z(dpsi)
        * lunar.ecliptic_of_date_au;
    let topocentric = astro::frames::geocentric_to_topocentric(geocentric, observer, sidereal);
    let geo_horizontal = astro::frames::equatorial_to_horizontal(geocentric, observer, sidereal);
    let topo_horizontal = astro::frames::equatorial_to_horizontal(topocentric, observer, sidereal);
    let parallax_arcminutes = horizontal_separation_arcsec(
        geo_horizontal.0.value(),
        geo_horizontal.1.value(),
        topo_horizontal.0.value(),
        topo_horizontal.1.value(),
    ) / 60.0;

    let result = PyDict::new_bound(py);
    result.set_item("gmst_max_seconds", gmst_max_seconds)?;
    result.set_item("gmst_samples", gmst_samples)?;
    result.set_item(
        "refraction_error_arcminutes",
        (sidera_refraction_arcminutes - bennett_refraction_arcminutes).abs(),
    )?;
    result.set_item("refraction_sidera_arcminutes", sidera_refraction_arcminutes)?;
    result.set_item(
        "refraction_bennett_arcminutes",
        bennett_refraction_arcminutes,
    )?;
    result.set_item("precession_arcminutes", precession_max)?;
    result.set_item("precession_median_arcminutes", precession_median)?;
    result.set_item("precession_min_arcminutes", precession_min)?;
    result.set_item(
        "precession_star_count",
        astro::catalog::bright_star_catalog()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .len(),
    )?;
    result.set_item("lunar_parallax_arcminutes", parallax_arcminutes)?;
    Ok(result.unbind())
}

fn horizontal_separation_arcsec(az_a: f64, alt_a: f64, az_b: f64, alt_b: f64) -> f64 {
    let delta_azimuth = (az_a - az_b).to_radians();
    let (alt_a, alt_b) = (alt_a.to_radians(), alt_b.to_radians());
    (alt_a.sin() * alt_b.sin() + alt_a.cos() * alt_b.cos() * delta_azimuth.cos())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
        * 3_600.0
}

#[pyfunction]
#[pyo3(signature = (width=768, height=512, certificate=None))]
pub(crate) fn _astro_night_golden_frame(
    py: Python<'_>,
    width: u32,
    height: u32,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<crate::py_types::frame::Frame>> {
    let capture = crate::core::certificate::begin_render_capture("_astro_night_golden_frame");
    let utc = UtcDateTime::new(2026, 12, 27, 14, 0, 0.0)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let observer = Observer::new(Angle::new(19.8207), Angle::new(-155.4681), 4_205.0)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let observation = astro::observation::set_observation(utc, observer)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let ctx = crate::core::gpu::try_ctx()?;
    let texture = astro::night_gpu::render_test_frame(
        ctx.device.clone(),
        ctx.queue.clone(),
        ctx.adapter.get_info().backend,
        &observation,
        width,
        height,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let frame = Py::new(
        py,
        crate::py_types::frame::Frame::new(
            ctx.device.clone(),
            ctx.queue.clone(),
            texture,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        ),
    )?;
    capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(frame)
}
