// src/labels/py_bindings.rs
// PyO3 bindings for label style types (P2.3) and the CARTOGRAPHER-PRIME
// optimal-declutter rationale.
// Exposes LabelStyle, LabelFlags, and LabelRationale to Python

use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
use pyo3::types::PyDict;

#[cfg(feature = "extension-module")]
use super::optimal::RationaleRecord;
use super::types::{LabelFlags, LabelStyle};

/// Internal scoped reservation used by `_DepthOcclusionSampler`. It is not
/// re-exported from the public `forge3d` package.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "_LabelDepthHostAllocation")]
pub struct PyLabelDepthHostAllocation {
    pub(crate) reservation: super::optimal::DepthHostAllocationReservation,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelDepthHostAllocation {
    #[getter]
    fn bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    #[getter]
    fn active(&self) -> bool {
        self.reservation.is_active()
    }

    /// Release now; returns true exactly once.
    fn close(&mut self) -> bool {
        self.reservation.release()
    }

    fn __repr__(&self) -> String {
        format!(
            "_LabelDepthHostAllocation(bytes={}, active={})",
            self.reservation.bytes(),
            self.reservation.is_active()
        )
    }
}

/// Python wrapper for LabelFlags.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LabelFlags")]
#[derive(Clone)]
pub struct PyLabelFlags {
    #[pyo3(get, set)]
    pub underline: bool,
    #[pyo3(get, set)]
    pub small_caps: bool,
    #[pyo3(get, set)]
    pub leader: bool,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelFlags {
    #[new]
    #[pyo3(signature = (underline=false, small_caps=false, leader=false))]
    fn new(underline: bool, small_caps: bool, leader: bool) -> Self {
        Self {
            underline,
            small_caps,
            leader,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LabelFlags(underline={}, small_caps={}, leader={})",
            self.underline, self.small_caps, self.leader
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<LabelFlags> for PyLabelFlags {
    fn from(f: LabelFlags) -> Self {
        Self {
            underline: f.underline,
            small_caps: f.small_caps,
            leader: f.leader,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<&PyLabelFlags> for LabelFlags {
    fn from(f: &PyLabelFlags) -> Self {
        Self {
            underline: f.underline,
            small_caps: f.small_caps,
            leader: f.leader,
        }
    }
}

/// Python wrapper for LabelStyle.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LabelStyle")]
#[derive(Clone)]
pub struct PyLabelStyle {
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub halo_color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub halo_width: f32,
    #[pyo3(get, set)]
    pub priority: i32,
    #[pyo3(get, set)]
    pub min_depth: f32,
    #[pyo3(get, set)]
    pub max_depth: f32,
    #[pyo3(get, set)]
    pub depth_fade: f32,
    #[pyo3(get, set)]
    pub min_zoom: f32,
    #[pyo3(get, set)]
    pub max_zoom: f32,
    #[pyo3(get, set)]
    pub rotation: f32,
    #[pyo3(get, set)]
    pub offset: (f32, f32),
    #[pyo3(get, set)]
    pub flags: PyLabelFlags,
    #[pyo3(get, set)]
    pub horizon_fade_angle: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelStyle {
    #[new]
    #[pyo3(signature = (
        size = 14.0,
        color = (0.1, 0.1, 0.1, 1.0),
        halo_color = (1.0, 1.0, 1.0, 0.8),
        halo_width = 1.5,
        priority = 0,
        min_depth = 0.0,
        max_depth = 1.0,
        depth_fade = 0.0,
        min_zoom = 0.0,
        max_zoom = 3.4028235e38,
        rotation = 0.0,
        offset = (0.0, 0.0),
        flags = None,
        horizon_fade_angle = 5.0,
    ))]
    #[allow(clippy::too_many_arguments)] // PyO3 constructor requires flat kwargs
    fn new(
        size: f32,
        color: (f32, f32, f32, f32),
        halo_color: (f32, f32, f32, f32),
        halo_width: f32,
        priority: i32,
        min_depth: f32,
        max_depth: f32,
        depth_fade: f32,
        min_zoom: f32,
        max_zoom: f32,
        rotation: f32,
        offset: (f32, f32),
        flags: Option<PyLabelFlags>,
        horizon_fade_angle: f32,
    ) -> Self {
        Self {
            size,
            color,
            halo_color,
            halo_width,
            priority,
            min_depth,
            max_depth,
            depth_fade,
            min_zoom,
            max_zoom,
            rotation,
            offset,
            flags: flags.unwrap_or_else(|| PyLabelFlags::new(false, false, false)),
            horizon_fade_angle,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LabelStyle(size={}, color={:?}, priority={}, halo_width={})",
            self.size, self.color, self.priority, self.halo_width
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<LabelStyle> for PyLabelStyle {
    fn from(s: LabelStyle) -> Self {
        Self {
            size: s.size,
            color: (s.color[0], s.color[1], s.color[2], s.color[3]),
            halo_color: (
                s.halo_color[0],
                s.halo_color[1],
                s.halo_color[2],
                s.halo_color[3],
            ),
            halo_width: s.halo_width,
            priority: s.priority,
            min_depth: s.min_depth,
            max_depth: s.max_depth,
            depth_fade: s.depth_fade,
            min_zoom: s.min_zoom,
            max_zoom: s.max_zoom,
            rotation: s.rotation,
            offset: (s.offset[0], s.offset[1]),
            flags: PyLabelFlags::from(s.flags),
            horizon_fade_angle: s.horizon_fade_angle,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<&PyLabelStyle> for LabelStyle {
    fn from(s: &PyLabelStyle) -> Self {
        Self {
            size: s.size,
            color: [s.color.0, s.color.1, s.color.2, s.color.3],
            halo_color: [
                s.halo_color.0,
                s.halo_color.1,
                s.halo_color.2,
                s.halo_color.3,
            ],
            halo_width: s.halo_width,
            priority: s.priority,
            min_depth: s.min_depth,
            max_depth: s.max_depth,
            depth_fade: s.depth_fade,
            min_zoom: s.min_zoom,
            max_zoom: s.max_zoom,
            rotation: s.rotation,
            offset: [s.offset.0, s.offset.1],
            flags: LabelFlags::from(&s.flags),
            horizon_fade_angle: s.horizon_fade_angle,
        }
    }
}

/// Overlap resolution for quantized areas: quantized-square units -> px^2.
#[cfg(feature = "extension-module")]
const AREA_SCALE: f64 = super::optimal::COORD_SCALE * super::optimal::COORD_SCALE;

#[cfg(feature = "extension-module")]
fn conflict_to_dict(py: Python<'_>, entry: &(u64, u32, i128)) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("label_id", entry.0)?;
    d.set_item("candidate_index", entry.1)?;
    d.set_item("overlap_area_px", entry.2 as f64 / AREA_SCALE)?;
    Ok(d.unbind())
}

#[cfg(feature = "extension-module")]
fn conflicts_to_text(entries: &[(u64, u32, i128)]) -> String {
    entries
        .iter()
        .map(|(label_id, candidate_index, area_q)| {
            format!(
                "label {} candidate {} (overlap {:.2} px^2)",
                label_id,
                candidate_index,
                *area_q as f64 / AREA_SCALE
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Grounded rationale for a bounded-optimal declutter solve.
///
/// Holds the typed decision records emitted by the solver; every rendered
/// line derives solely from these records (never a post-hoc narrative).
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LabelRationale")]
#[derive(Clone)]
pub struct PyLabelRationale {
    pub(crate) records: Vec<RationaleRecord>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelRationale {
    /// Typed decision records as a list of dicts (each with a "kind" key).
    fn records(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let weight_scale = super::optimal::WEIGHT_SCALE;
        self.records
            .iter()
            .map(|record| {
                let d = PyDict::new_bound(py);
                match record {
                    RationaleRecord::Placed {
                        label_id,
                        candidate_index,
                        weight_q,
                        displaced,
                    } => {
                        d.set_item("kind", "placed")?;
                        d.set_item("label_id", label_id)?;
                        d.set_item("candidate_index", candidate_index)?;
                        d.set_item("weight", *weight_q as f64 / weight_scale)?;
                        d.set_item(
                            "displaced",
                            displaced
                                .iter()
                                .map(|entry| conflict_to_dict(py, entry))
                                .collect::<PyResult<Vec<_>>>()?,
                        )?;
                    }
                    RationaleRecord::Dropped {
                        label_id,
                        candidate_index,
                        weight_q,
                        priority_lost,
                        blocking,
                    } => {
                        d.set_item("kind", "dropped")?;
                        d.set_item("label_id", label_id)?;
                        d.set_item("candidate_index", candidate_index)?;
                        d.set_item("weight", *weight_q as f64 / weight_scale)?;
                        d.set_item("priority_lost", priority_lost)?;
                        d.set_item(
                            "blocking",
                            blocking
                                .iter()
                                .map(|entry| conflict_to_dict(py, entry))
                                .collect::<PyResult<Vec<_>>>()?,
                        )?;
                    }
                    RationaleRecord::VisibilityFilteredCandidate {
                        label_id,
                        candidate_index,
                    } => {
                        d.set_item("kind", "visibility_filtered_candidate")?;
                        d.set_item("label_id", label_id)?;
                        d.set_item("candidate_index", candidate_index)?;
                    }
                    RationaleRecord::Solver {
                        nodes_explored,
                        certified,
                        budget_exhausted,
                        objective_q,
                        upper_bound_q,
                        gap,
                        gap_tolerance,
                    } => {
                        d.set_item("kind", "solver")?;
                        d.set_item("nodes_explored", nodes_explored)?;
                        d.set_item("certified", certified)?;
                        d.set_item("budget_exhausted", budget_exhausted)?;
                        d.set_item("objective", *objective_q as f64 / weight_scale)?;
                        d.set_item("upper_bound", *upper_bound_q as f64 / weight_scale)?;
                        d.set_item("gap", gap)?;
                        d.set_item("gap_tolerance", gap_tolerance)?;
                    }
                }
                Ok(d.unbind())
            })
            .collect()
    }

    /// Human-readable lines derived purely from the recorded decisions.
    fn render(&self) -> Vec<String> {
        self.records
            .iter()
            .map(|record| match record {
                RationaleRecord::Placed {
                    label_id,
                    candidate_index,
                    weight_q,
                    displaced,
                } => {
                    let mut line = format!(
                        "placed label {} at candidate {} (weight {:.3})",
                        label_id,
                        candidate_index,
                        *weight_q as f64 / super::optimal::WEIGHT_SCALE
                    );
                    if !displaced.is_empty() {
                        line.push_str("; displaced ");
                        line.push_str(&conflicts_to_text(displaced));
                    }
                    line
                }
                RationaleRecord::Dropped {
                    label_id,
                    candidate_index,
                    priority_lost,
                    blocking,
                    ..
                } => format!(
                    "dropped label {} candidate {} ({}): blocked by {}",
                    label_id,
                    candidate_index,
                    if *priority_lost { "priority_lost" } else { "collision" },
                    conflicts_to_text(blocking)
                ),
                RationaleRecord::VisibilityFilteredCandidate {
                    label_id,
                    candidate_index,
                } => format!(
                    "label {} candidate {} excluded by the caller-supplied visibility gate",
                    label_id, candidate_index
                ),
                RationaleRecord::Solver {
                    nodes_explored,
                    certified,
                    budget_exhausted,
                    objective_q,
                    upper_bound_q,
                    gap,
                    gap_tolerance,
                } => format!(
                    "solver: {} nodes explored, certified={}, budget_exhausted={}, objective={:.3}, upper_bound={:.3}, gap={:.6} (tolerance {:.6})",
                    nodes_explored,
                    certified,
                    budget_exhausted,
                    *objective_q as f64 / super::optimal::WEIGHT_SCALE,
                    *upper_bound_q as f64 / super::optimal::WEIGHT_SCALE,
                    gap,
                    gap_tolerance
                ),
            })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.records.len()
    }

    fn __repr__(&self) -> String {
        format!("LabelRationale({} records)", self.records.len())
    }
}
