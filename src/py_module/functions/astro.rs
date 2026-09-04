use super::*;

pub(super) fn register_astro_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(astro_body_position, m)?)?;
    m.add_function(wrap_pyfunction!(astro_moon_phase, m)?)?;
    m.add_function(wrap_pyfunction!(astro_delta_t_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(astro_sidereal_time, m)?)?;
    m.add_function(wrap_pyfunction!(astro_refraction_arcminutes, m)?)?;
    m.add_function(wrap_pyfunction!(sky_set_observation, m)?)?;
    m.add_function(wrap_pyfunction!(astro_validation_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(_astro_night_golden_frame, m)?)?;
    Ok(())
}
