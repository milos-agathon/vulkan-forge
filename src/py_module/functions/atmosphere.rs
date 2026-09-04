use super::*;

pub(super) fn register_atmosphere_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(atmosphere_bake_luts, m)?)?;
    m.add_function(wrap_pyfunction!(atmosphere_spectral_to_linear_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(atmosphere_generate_environment, m)?)?;
    m.add_function(wrap_pyfunction!(atmosphere_reference_aerial, m)?)?;
    Ok(())
}
