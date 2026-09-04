use super::*;
use crate::vector::api::{PolygonDef, PolylineDef, VectorStyle};
use crate::vector::coverage::{
    render_compiled_coverage, CoverageGeometry, CoverageGeometryBuilder, FillRule, VectorQuality,
};
use numpy::PyArray1;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoverageSceneInput {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) layers: Vec<CoverageLayerInput>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoverageLayerInput {
    pub(super) name: String,
    pub(super) quality: String,
    pub(super) fill_rule: String,
    pub(super) color: [f32; 4],
    #[serde(default)]
    pub(super) polygons: Vec<CoveragePolygonInput>,
    #[serde(default)]
    pub(super) polylines: Vec<CoveragePolylineInput>,
    pub(super) polygon_grid: Option<CoverageGridInput>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoveragePolygonInput {
    pub(super) exterior: Vec<[f32; 2]>,
    #[serde(default)]
    pub(super) holes: Vec<Vec<[f32; 2]>>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoveragePolylineInput {
    pub(super) path: Vec<[f32; 2]>,
    pub(super) width: f32,
    pub(super) cap: String,
    pub(super) join: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoverageGridInput {
    pub(super) columns: u32,
    pub(super) rows: u32,
    pub(super) origin: [f32; 2],
    pub(super) cell_size: [f32; 2],
}

pub(super) fn decode_coverage_scene(scene_json: &str) -> Result<CoverageSceneInput, RenderError> {
    serde_json::from_str(scene_json)
        .map_err(|error| RenderError::Upload(format!("vector_coverage_scene_json: {error}")))
}

pub(super) fn parse_coverage_scene(scene_json: &str) -> Result<CoverageGeometry, RenderError> {
    let input = decode_coverage_scene(scene_json)?;
    ingest_coverage_scene(input)
}

pub(super) fn ingest_coverage_scene(
    input: CoverageSceneInput,
) -> Result<CoverageGeometry, RenderError> {
    let mut builder = CoverageGeometryBuilder::new(input.width, input.height)?;
    for layer in input.layers {
        let quality = VectorQuality::parse(&layer.quality).ok_or_else(|| {
            RenderError::Upload(format!(
                "vector_coverage_quality: expected 'default' or 'analytic', got {:?}",
                layer.quality
            ))
        })?;
        if quality != VectorQuality::Analytic {
            return Err(RenderError::Upload(format!(
                "vector_coverage_quality: layer {:?} is {:?}; the analytic entry point \
                 is opt-in and will not replace the default pipeline",
                layer.name, layer.quality
            )));
        }
        let fill_rule = FillRule::parse(&layer.fill_rule).ok_or_else(|| {
            RenderError::Upload(format!(
                "vector_coverage_fill_rule: expected 'nonzero' or 'evenodd', got {:?}",
                layer.fill_rule
            ))
        })?;
        let layer_id = builder.add_layer(layer.name, fill_rule, layer.color)?;
        for polygon in layer.polygons {
            builder.push_polygon(layer_id, &polygon.into_polygon())?;
        }
        if let Some(grid) = layer.polygon_grid {
            grid.push_polygons(layer_id, &mut builder)?;
        }
        for polyline in layer.polylines {
            if polyline.cap != "round" || polyline.join != "round" {
                return Err(RenderError::Upload(format!(
                    "vector_coverage_stroke_style: LIMES currently requires round cap/join, \
                     got cap={:?} join={:?}",
                    polyline.cap, polyline.join
                )));
            }
            builder.push_round_polyline(layer_id, &polyline.into_polyline())?;
        }
    }
    builder.finish()
}

impl CoveragePolygonInput {
    fn into_polygon(self) -> PolygonDef {
        PolygonDef {
            exterior: points(self.exterior),
            holes: self.holes.into_iter().map(points).collect(),
            style: VectorStyle::default(),
        }
    }
}

impl CoveragePolylineInput {
    fn into_polyline(self) -> PolylineDef {
        PolylineDef {
            path: points(self.path),
            style: VectorStyle {
                stroke_width: self.width,
                ..VectorStyle::default()
            },
        }
    }
}

impl CoverageGridInput {
    fn push_polygons(
        self,
        layer: u32,
        builder: &mut CoverageGeometryBuilder,
    ) -> Result<(), RenderError> {
        if self.columns == 0
            || self.rows == 0
            || !self
                .cell_size
                .into_iter()
                .all(|size| size.is_finite() && size > 0.0)
        {
            return Err(RenderError::Upload(
                "vector_coverage_grid: rows, columns, and cell sizes must be positive".into(),
            ));
        }
        for row in 0..self.rows {
            for column in 0..self.columns {
                let [column, row] = viewport_dims(column, row);
                let x = self.origin[0] + column * self.cell_size[0];
                let y = self.origin[1] + row * self.cell_size[1];
                builder.push_polygon(
                    layer,
                    &PolygonDef {
                        exterior: points(vec![
                            [x, y],
                            [x + self.cell_size[0], y],
                            [x + self.cell_size[0], y + self.cell_size[1]],
                            [x, y + self.cell_size[1]],
                        ]),
                        holes: Vec::new(),
                        style: VectorStyle::default(),
                    },
                )?;
            }
        }
        Ok(())
    }
}

fn points(values: Vec<[f32; 2]>) -> Vec<glam::Vec2> {
    values
        .into_iter()
        .map(|[x, y]| glam::Vec2::new(x, y))
        .collect()
}

#[cfg(feature = "extension-module")]
#[pyfunction(signature = (scene_json, include_coverage=false, include_records=false, certificate=None))]
pub(crate) fn vector_render_analytic_py(
    py: Python<'_>,
    scene_json: &str,
    include_coverage: bool,
    include_records: bool,
    certificate: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let capture = crate::core::certificate::begin_render_capture("vector_render_analytic_py");
    let mut lookup = super::coverage_cache::compiled_coverage_scene(scene_json)?;
    let compiled = lookup.compiled.clone();
    let geometry = compiled.geometry();
    let (device, queue) = gpu_device_queue()?;
    let output = render_compiled_coverage(&device, &queue, &compiled)?;
    lookup.commit()?;
    capture.finish();
    let execution_report_json = crate::core::certificate::execution_report_json()?;
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;

    let result = PyDict::new_bound(py);
    let report = coverage_report_py(py, &geometry, &output, &execution_report_json)?;
    let rgba = PyArray1::<u8>::from_vec_bound(py, output.rgba8).reshape([
        geometry.height as usize,
        geometry.width as usize,
        4,
    ])?;
    result.set_item("rgba", rgba)?;
    if include_coverage {
        let coverage = PyArray1::<f32>::from_vec_bound(py, output.coverage.clone()).reshape([
            geometry.layers.len(),
            geometry.height as usize,
            geometry.width as usize,
        ])?;
        result.set_item("coverage", coverage)?;
    }
    if include_records {
        result.set_item("records", primitive_records_py(py, &geometry)?)?;
    }
    result.set_item("report", report)?;
    Ok(result.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub(crate) fn vector_coverage_primitives_py(
    py: Python<'_>,
    scene_json: &str,
) -> PyResult<Py<PyAny>> {
    let geometry = parse_coverage_scene(scene_json)?;
    let result = PyDict::new_bound(py);
    result.set_item("width", geometry.width)?;
    result.set_item("height", geometry.height)?;
    result.set_item(
        "fill_rules",
        geometry
            .layers
            .iter()
            .map(|layer| layer.fill_rule.as_str())
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "colors",
        geometry
            .layers
            .iter()
            .map(|layer| layer.color)
            .collect::<Vec<_>>(),
    )?;
    result.set_item("records", primitive_records_py(py, &geometry)?)?;
    Ok(result.into_py(py))
}

fn primitive_records_py<'py>(
    py: Python<'py>,
    geometry: &CoverageGeometry,
) -> PyResult<Bound<'py, PyList>> {
    let records = PyList::empty_bound(py);
    for record in &geometry.primitives {
        let value = PyDict::new_bound(py);
        value.set_item("kind", record.kind() as u32)?;
        value.set_item("geometry", record.geometry)?;
        value.set_item("bounds", record.bounds)?;
        value.set_item("layer", record.layer())?;
        value.set_item("winding", record.winding())?;
        value.set_item("stable_id", record.stable_id())?;
        records.append(value)?;
    }
    Ok(records)
}

fn coverage_report_py<'py>(
    py: Python<'py>,
    geometry: &CoverageGeometry,
    output: &crate::vector::coverage::CoverageRenderOutput,
    execution_report_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let report = PyDict::new_bound(py);
    report.set_item("quality", "analytic")?;
    report.set_item("width", geometry.width)?;
    report.set_item("height", geometry.height)?;
    report.set_item("layer_count", geometry.layers.len())?;
    report.set_item("primitive_count", output.stats.primitive_count)?;
    report.set_item("line_count", output.stats.line_count)?;
    report.set_item("arc_count", output.stats.arc_count)?;
    report.set_item("tile_columns", output.stats.tile_columns)?;
    report.set_item("tile_rows", output.stats.tile_rows)?;
    report.set_item("tile_capacity", output.stats.tile_capacity)?;
    report.set_item("measured_memberships", output.stats.measured_memberships)?;
    report.set_item("written_memberships", output.stats.written_memberships)?;
    report.set_item("populated_tiles", output.stats.populated_tiles)?;
    report.set_item("active_pixel_count", output.stats.active_pixel_count)?;
    report.set_item("resolve_pixel_count", output.stats.resolve_pixel_count)?;
    report.set_item("dispatch_retries", output.stats.dispatch_retries)?;
    report.set_item("allocation_bytes", output.stats.allocation_bytes)?;
    report.set_item("wall_ms", output.stats.wall_ms)?;
    report.set_item("output_sha256", &output.stats.output_sha256)?;
    report.set_item("structured_errors", output.errors)?;
    report.set_item("execution_report_json", execution_report_json)?;

    let pixel_count = geometry.width as usize * geometry.height as usize;
    let layers = PyList::empty_bound(py);
    for (index, layer) in geometry.layers.iter().enumerate() {
        let values = &output.coverage[index * pixel_count..(index + 1) * pixel_count];
        let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / pixel_count as f64;
        let stats = PyDict::new_bound(py);
        stats.set_item("name", &layer.name)?;
        stats.set_item("fill_rule", layer.fill_rule.as_str())?;
        stats.set_item("minimum", minimum)?;
        stats.set_item("maximum", maximum)?;
        stats.set_item("mean", mean)?;
        stats.set_item(
            "nonzero_pixels",
            values.iter().filter(|&&value| value > 0.0).count(),
        )?;
        layers.append(stats)?;
    }
    report.set_item("layers", layers)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_quality_is_explicit_and_default_is_rejected_by_analytic_entry() {
        let scene = r#"{
            "width": 4,
            "height": 4,
            "layers": [{
                "name": "default",
                "quality": "default",
                "fill_rule": "nonzero",
                "color": [1.0, 1.0, 1.0, 1.0],
                "polygons": [],
                "polylines": []
            }]
        }"#;
        assert!(parse_coverage_scene(scene).is_err());
    }

    #[test]
    fn committed_grid_schema_materializes_one_hundred_polygons() {
        let scene = r#"{
            "width": 20,
            "height": 20,
            "layers": [{
                "name": "mosaic",
                "quality": "analytic",
                "fill_rule": "nonzero",
                "color": [1.0, 1.0, 1.0, 1.0],
                "polygon_grid": {
                    "columns": 10,
                    "rows": 10,
                    "origin": [0.0, 0.0],
                    "cell_size": [2.0, 2.0]
                }
            }]
        }"#;
        let geometry = parse_coverage_scene(scene).unwrap();
        assert_eq!(geometry.primitives.len(), 400);
    }

    #[test]
    fn round_stroke_schema_materializes_exact_arcs() {
        let scene = r#"{
            "width": 16,
            "height": 16,
            "layers": [{
                "name": "road",
                "quality": "analytic",
                "fill_rule": "nonzero",
                "color": [1.0, 0.0, 0.0, 1.0],
                "polylines": [{
                    "path": [[1.0, 1.0], [15.0, 15.0]],
                    "width": 0.5,
                    "cap": "round",
                    "join": "round"
                }]
            }]
        }"#;
        let geometry = parse_coverage_scene(scene).unwrap();
        assert!(geometry
            .primitives
            .iter()
            .any(|record| record.kind() == crate::vector::coverage::PrimitiveKind::Arc));
    }
}
