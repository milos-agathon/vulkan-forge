use super::*;

#[cfg(feature = "extension-module")]
pub(crate) fn register_geodesy_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_functions::solar_position, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::terrain_viewshed, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::terrain_shadow_mask,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::terrain_shadow_tip,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::body_info, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::areoid_undulation, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::geoid_undulation, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::orthometric_to_ellipsoidal,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::ellipsoidal_to_orthometric,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::geodesic_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::geodesic_direct, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::wgs84_to_ecef, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_functions::ecef_to_wgs84, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_functions::dem_orthometric_to_ellipsoidal,
        m
    )?)?;
    Ok(())
}
