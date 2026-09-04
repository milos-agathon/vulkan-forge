use super::super::*;
use crate::py_functions::*;

mod anamnesis;
mod astro;
mod atmosphere;
mod camera;
mod codec;
mod diagnostics;
mod geodesy;
mod geometry;
mod gis;
mod interactive;
mod io_import;
mod labels;
mod license;
mod precision;
mod provenance;
mod rendering;
mod tiles3d;

#[cfg(feature = "extension-module")]
pub(crate) fn register_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    anamnesis::register_anamnesis_py_functions(m)?;
    atmosphere::register_atmosphere_py_functions(m)?;
    astro::register_astro_py_functions(m)?;
    interactive::register_interactive_py_functions(m)?;
    geometry::register_geometry_py_functions(m)?;
    io_import::register_io_import_py_functions(m)?;
    diagnostics::register_diagnostics_py_functions(m)?;
    license::register_license_py_functions(m)?;
    provenance::register_provenance_py_functions(m)?;
    precision::register_precision_py_functions(m)?;
    camera::register_camera_py_functions(m)?;
    codec::register_codec_py_functions(m)?;
    rendering::register_rendering_py_functions(m)?;
    gis::register_gis_py_functions(m)?;
    geodesy::register_geodesy_py_functions(m)?;
    tiles3d::register_tiles3d_py_functions(m)?;
    labels::register_labels_py_functions(m)?;
    Ok(())
}
