//! LIMES analytic-coverage vector rasterization.
//!
//! This opt-in path consumes polygon rings before lyon tessellation and
//! polyline centerlines before the legacy quad expansion.  Geometry is kept as
//! directed linear edges and y-monotone circular arcs so the raster kernel can
//! integrate coverage rather than approximate it with samples.

mod binning;
mod ingest;
mod math;
mod raster;
mod render;
mod resolve;
mod types;

#[cfg(test)]
mod math_tests;
#[cfg(test)]
mod raster_tests;
#[cfg(test)]
mod resolve_tests;

pub use binning::{BinLayout, CoverageBinner, CoverageBins};
pub use ingest::CoverageGeometryBuilder;
pub use math::{analytic_coverage_pixel, circle_pixel_intersection_area, rasterize_coverage_cpu};
pub use raster::{CoverageRasterResources, CoverageRasterizer};
pub(crate) use render::{compile_coverage, render_compiled_coverage, CoverageCompiledScene};
pub use render::{render_coverage, CoverageRenderOutput, CoverageRenderStats};
pub use resolve::{CoverageResolveResources, CoverageResolver};
pub use types::{
    CoverageGeometry, CoverageLayer, FillRule, PrimitiveKind, PrimitiveRecord, VectorQuality,
    COVERAGE_TILE_SIZE,
};
