use super::*;

pub(super) fn register_diagnostics_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enumerate_adapters, m)?)?;
    m.add_function(wrap_pyfunction!(device_probe, m)?)?;
    m.add_function(wrap_pyfunction!(global_memory_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(set_memory_budget_policy, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_budget_policy, m)?)?;
    m.add_function(wrap_pyfunction!(
        request_host_visible_allocation_for_test,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(render_debug_pattern_frame, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_to_exr, m)?)?;

    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(report_device, m)?)?;
    m.add_function(wrap_pyfunction!(c5_build_framegraph_report, m)?)?;
    m.add_function(wrap_pyfunction!(c6_mt_record_demo, m)?)?;
    m.add_function(wrap_pyfunction!(c7_async_compute_demo, m)?)?;
    m.add_function(wrap_pyfunction!(native_degradations, m)?)?;
    m.add_function(wrap_pyfunction!(terrain_culling_stats, m)?)?;
    m.add_function(wrap_pyfunction!(terrain_visibility_stats, m)?)?;
    m.add_function(wrap_pyfunction!(terrain_vt_stats, m)?)?;
    m.add_function(wrap_pyfunction!(terrain_seam_stats, m)?)?;
    m.add_function(wrap_pyfunction!(clear_native_degradations, m)?)?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(render_execution_report, m)?)?;
    m.add_function(wrap_pyfunction!(
        _record_terrain_poster_certificate_inputs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(begin_render_execution_capture, m)?)?;
    m.add_function(wrap_pyfunction!(finish_render_execution_capture, m)?)?;
    m.add_function(wrap_pyfunction!(abort_render_execution_capture, m)?)?;
    m.add_function(wrap_pyfunction!(sign_render_certificate_digest, m)?)?;
    m.add_function(wrap_pyfunction!(shader_report, m)?)?;
    Ok(())
}
