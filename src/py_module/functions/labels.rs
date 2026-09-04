use super::*;

pub(super) fn register_labels_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::labels::py_text::PyShapedText>()?;
    m.add_function(wrap_pyfunction!(reserve_label_depth_host_allocation_py, m)?)?;
    m.add_function(wrap_pyfunction!(declutter_py, m)?)?;
    m.add_function(wrap_pyfunction!(declutter_optimal_py, m)?)?;
    m.add_function(wrap_pyfunction!(text_shape_py, m)?)?;
    m.add_function(wrap_pyfunction!(rasterize_shaped_run_py, m)?)?;
    m.add_function(wrap_pyfunction!(bake_msdf_atlas_py, m)?)?;
    m.add_function(wrap_pyfunction!(bake_msdf_atlas_shaped_py, m)?)?;
    Ok(())
}
