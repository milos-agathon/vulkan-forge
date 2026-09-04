// src/lib.rs
// Rust crate root for forge3d - GPU rendering library with Python bindings
// Provides SDF primitives, CSG operations, hybrid traversal, and path tracing
// RELEVANT FILES:src/sdf/mod.rs,src/path_tracing/mod.rs,python/forge3d/__init__.py

#[cfg(feature = "extension-module")]
pub(crate) use once_cell::sync::Lazy;
#[cfg(feature = "extension-module")]
pub(crate) use std::sync::Mutex;

#[cfg(feature = "extension-module")]
pub(crate) use shadows::state::{CpuCsmConfig, CpuCsmState};

#[cfg(feature = "extension-module")]
pub(crate) use glam::Vec3;
#[cfg(feature = "extension-module")]
pub(crate) use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArrayDyn};
#[cfg(feature = "extension-module")]
pub(crate) use pyo3::types::{PyByteArray, PyBytes, PyDict, PyMemoryView, PyString};
#[cfg(feature = "extension-module")]
pub(crate) use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    pyclass, pyfunction, pymethods, wrap_pyfunction,
};

// C1/C3/C5/C6/C7: Additional imports for PyO3 functions
#[cfg(feature = "extension-module")]
pub(crate) use crate::core::async_compute::{
    AsyncComputeConfig as AcConfig, AsyncComputeScheduler as AcScheduler,
    ComputePassDescriptor as AcPassDesc, DispatchParams as AcDispatch,
};
#[cfg(feature = "extension-module")]
pub(crate) use crate::core::context as engine_context;
#[cfg(feature = "extension-module")]
pub(crate) use crate::core::device_caps::DeviceCaps;
#[cfg(feature = "extension-module")]
pub(crate) use crate::core::multi_thread::{
    CopyTask as MtCopyTask, MultiThreadConfig as MtConfig, MultiThreadRecorder as MtRecorder,
};
#[cfg(feature = "extension-module")]
pub(crate) use crate::renderer::readback::read_texture_tight;
#[cfg(feature = "extension-module")]
pub(crate) use crate::sdf::hybrid::Ray as HybridRay;
#[cfg(feature = "extension-module")]
pub(crate) use crate::sdf::py::{PySdfPrimitive, PySdfScene, PySdfSceneBuilder};
#[cfg(all(feature = "extension-module", feature = "images"))]
pub(crate) use crate::util::exr_write;
#[cfg(feature = "extension-module")]
pub(crate) use crate::util::image_write;
#[cfg(feature = "extension-module")]
pub(crate) use std::path::Path;
#[cfg(feature = "extension-module")]
pub(crate) use std::sync::Arc;
#[cfg(feature = "extension-module")]
static GLOBAL_CSM_STATE: Lazy<Mutex<CpuCsmState>> =
    Lazy::new(|| Mutex::new(CpuCsmState::default()));

#[cfg(feature = "extension-module")]
mod py_functions;
#[cfg(feature = "extension-module")]
mod py_module;
#[cfg(feature = "extension-module")]
mod py_types;
#[cfg(feature = "extension-module")]
pub use py_types::*;
pub mod math {
    /// Orthonormalize a tangent `t` against normal `n` and return (tangent, bitangent).
    ///
    /// Uses simple Gram-Schmidt then computes bitangent as cross(n, t_ortho).
    pub fn orthonormalize_tangent(n: [f32; 3], t: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }

        fn norm(v: [f32; 3]) -> f32 {
            dot(v, v).sqrt()
        }
        fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        fn mul(v: [f32; 3], s: f32) -> [f32; 3] {
            [v[0] * s, v[1] * s, v[2] * s]
        }
        fn normalize(v: [f32; 3]) -> [f32; 3] {
            let l = norm(v);
            if l > 0.0 {
                [v[0] / l, v[1] / l, v[2] / l]
            } else {
                v
            }
        }
        fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }

        let n_n = normalize(n);
        let t_ortho = normalize(sub(t, mul(n_n, dot(n_n, t))));
        let b = cross(n_n, t_ortho);
        (t_ortho, b)
    }
}

// Rendering modules
pub mod accel;
pub mod astro;
pub mod camera;
pub mod cli;
pub mod codec;
pub mod colormap;
pub mod converters; // Geometry converters (e.g., MultipolygonZ -> OBJ)
pub mod core;
pub mod external_image;
pub mod formats;
pub mod geo; // P3-reproject: Geographic utilities (CRS reprojection)
pub mod geometry;
pub mod gis; // Rust-backed GIS raster metadata and GeoTIFF write utilities
pub mod import; // Importers: OSM buildings, etc.
pub mod io; // IO: OBJ/PLY/glTF readers/writers
pub mod lighting; // P0: Production-ready lighting stack (lights, BRDFs, shadows, IBL)
pub mod loaders;
pub mod mesh;
pub mod offscreen; // P7: Offscreen PBR harness for BRDF galleries and CI goldens
pub mod path_tracing;
pub mod pipeline;
pub mod render; // Rendering utilities (instancing)
pub mod renderer;
#[cfg(feature = "extension-module")]
pub mod scene;
pub mod sdf; // New SDF module
pub(crate) mod shader_sources;
pub mod shadows; // Shadow mapping implementations
pub mod smoke; // Physical smoke volumes, simulation, and reference rendering
pub mod terrain;
pub mod uv; // UV unwrap helpers (planar, spherical)
pub mod textures {}
pub mod animation; // Feature C: Camera animation and keyframe interpolation
pub mod bundle; // Scene bundle (.forge3d) for portable scene packages
pub mod export;
pub mod labels; // Screen-space text labels with MSDF rendering
pub mod license; // Ed25519 offline license signature verification
pub mod p5;
pub mod passes;
pub mod picking; // Feature picking and inspection system
pub mod pointcloud; // P5: Point Cloud support (COPC, EPT)
pub mod style; // Mapbox Style Spec import for vector/label styling
pub mod tiles3d; // P5: 3D Tiles support (tileset.json, b3dm, pnts)
pub mod util;
pub mod vector;
pub mod verify;
pub mod viewer; // Interactive windowed viewer (Workstream I1) // P5.2: render passes wrappers // P5-export: Vector export (SVG/PDF) for print-grade overlays

// Re-export commonly used types
pub use core::cloud_shadows::{
    CloudAnimationParams, CloudShadowQuality, CloudShadowRenderer, CloudShadowUniforms,
};
pub use core::clouds::{
    CloudAnimationPreset, CloudInstance, CloudParams, CloudQuality, CloudRenderMode, CloudRenderer,
    CloudUniforms,
};
pub use core::dof::{CameraDofParams, DofMethod, DofQuality, DofRenderer, DofUniforms};
pub use core::dual_source_oit::{
    DualSourceComposeUniforms, DualSourceOITMode, DualSourceOITQuality, DualSourceOITRenderer,
    DualSourceOITStats, DualSourceOITUniforms,
};
pub use core::error::RenderError;
pub use core::ground_plane::{
    GroundPlaneMode, GroundPlaneParams, GroundPlaneRenderer, GroundPlaneUniforms,
};
pub use core::ibl::{IBLQuality, IBLRenderer};
pub use core::ltc_area_lights::{LTCRectAreaLightRenderer, LTCUniforms, RectAreaLight};
pub use core::point_spot_lights::{
    DebugMode, Light, LightPreset, LightType, PointSpotLightRenderer, PointSpotLightUniforms,
    ShadowQuality,
};
pub use core::reflections::{PlanarReflectionRenderer, ReflectionQuality};
pub use core::soft_light_radius::{
    SoftLightFalloffMode, SoftLightPreset, SoftLightRadiusRenderer, SoftLightRadiusUniforms,
};
pub use core::water_surface::{
    WaterSurfaceMode, WaterSurfaceParams, WaterSurfaceRenderer, WaterSurfaceUniforms,
};
pub use lighting::LightBuffer;
pub use path_tracing::{TracerEngine, TracerParams};
pub use render::params::{
    AtmosphereParams as RendererAtmosphereParams, BrdfModel as RendererBrdfModel,
    ConfigError as RendererConfigError, GiMode as RendererGiMode, GiParams as RendererGiParams,
    LightConfig as RendererLightConfig, LightType as RendererLightType,
    LightingParams as RendererLightingParams, RendererConfig, ShadowParams as RendererShadowParams,
    ShadowTechnique as RendererShadowTechnique, SkyModel as RendererSkyModel,
    SsrParams as RendererSsrParams, VolumetricParams as RendererVolumetricParams,
    VolumetricPhase as RendererVolumetricPhase,
};
pub use sdf::{
    CsgOperation, HybridHitResult, HybridScene, SdfPrimitive, SdfPrimitiveType, SdfScene,
    SdfSceneBuilder,
};
pub use shadows::{
    detect_peter_panning, CascadeStatistics, CascadedShadowMaps, CsmConfig, CsmRenderer,
    ShadowManager, ShadowManagerConfig,
};

// PyO3 module entry point so Python can `import forge3d._forge3d`
// This must be named exactly `_forge3d` to match [tool.maturin].module-name in pyproject.toml
#[cfg(feature = "extension-module")]
#[pymodule]
fn _forge3d(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "forge3d native module")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Typed GPU-error exceptions so Python callers can catch budget/degradation
    // failures precisely (see src/core/error.rs and to_py_err mapping).
    m.add(
        "MemoryBudgetExceeded",
        _py.get_type_bound::<crate::core::error::MemoryBudgetExceeded>(),
    )?;
    m.add(
        "DegradedCapability",
        _py.get_type_bound::<crate::core::error::DegradedCapability>(),
    )?;
    // MENSURA M-05: structured raster-reprojection failure with stable
    // .count/.first_pixel/.src_crs/.dst_crs/.policy attributes.
    m.add(
        "TransformFailed",
        _py.get_type_bound::<crate::gis::error::TransformFailed>(),
    )?;
    py_module::register_py_functions(m)?;
    py_module::register_py_classes(m)?;
    Ok(())
}
