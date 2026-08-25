use super::*;

pub(super) fn register_export_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::export::projection::project_3d_to_2d_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::export::projection::project_2d_to_screen_py,
        m
    )?)?;
    Ok(())
}
