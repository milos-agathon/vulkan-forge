use super::*;

pub(super) fn register_rendering_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::terrain::clipmap::py_bindings::register_clipmap_bindings(m)?;
    #[cfg(feature = "cog_streaming")]
    crate::terrain::cog::py_bindings::register_cog_bindings(m)?;

    m.add_function(wrap_pyfunction!(_pt_render_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_render_terrain_reference, m)?)?;
    m.add_function(wrap_pyfunction!(
        hybrid_render_aether_spectral_reference,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(render_adjudication_pair, m)?)?;
    m.add_function(wrap_pyfunction!(render_brdf_tile, m)?)?;
    m.add_function(wrap_pyfunction!(render_brdf_tile_overrides, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::lighting::py_bindings::sun_position,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::lighting::py_bindings::sun_position_utc,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(read_laz_points_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_laz_point_attributes_py, m)?)?;
    m.add_function(wrap_pyfunction!(copc_read_node_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(copc_laz_enabled_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geo::reproject::proj_available_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geo::reproject::reproject_coords_py,
        m
    )?)?;

    #[cfg(feature = "enable-tbn")]
    {
        m.add_function(wrap_pyfunction!(
            crate::mesh::tbn::mesh_generate_cube_tbn,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::mesh::tbn::mesh_generate_plane_tbn,
            m
        )?)?;
    }

    Ok(())
}
