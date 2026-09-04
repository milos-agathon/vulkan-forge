use super::*;

pub(super) fn register_codec_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_bc7_rgba8, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bc7_rgba8, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bc5_rg8, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bc5_rg8, m)?)?;
    m.add_function(wrap_pyfunction!(compress_dem, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_dem, m)?)?;
    m.add_function(wrap_pyfunction!(verify_dem, m)?)?;
    Ok(())
}
