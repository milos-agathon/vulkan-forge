//! Bounded-optimal label placement via deterministic branch-and-bound.
//!
//! CARTOGRAPHER-PRIME: maximizes total placed priority weight subject to
//! (a) at most one placed candidate per label, (b) no two placed boxes
//! overlapping, and (c) `visible` candidates only. All arithmetic that
//! affects branching is integer (fixed-precision quantized), the search
//! order is a pure function of the candidate total order
//! `(label_id, candidate_index)`, and every decision is recorded as a
//! typed [`RationaleRecord`] so the emitted rationale is grounded in the
//! actual geometric conflicts the solver resolved — never a post-hoc
//! narrative.

use std::collections::BTreeMap;
use std::fmt;

use super::declutter::{DeclutterConfig, DeclutterResult, PlacementCandidate};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_host_allocation, ResourceHandle};
use crate::core::text_overlay::TextInstance;

/// Fixed grid for box coordinates: 1/16 px, so floating-point drift cannot
/// change branch decisions across devices.
pub const COORD_SCALE: f64 = 16.0;
/// Fixed grid for priority weights: 1/1024 weight units.
pub const WEIGHT_SCALE: f64 = 1024.0;
/// Fixed grid for the relative optimality-gap tolerance.
const GAP_SCALE: i64 = 1_000_000_000;

/// Scoped reservation for CPU label-occlusion arrays used by label-plan
/// compilation. This owns the authoritative tracker handle; `release` and
/// `Drop` are idempotent because ownership is represented by `Option::take`.
#[derive(Debug)]
pub struct DepthHostAllocationReservation {
    handle: Option<ResourceHandle>,
    bytes: u64,
}

impl DepthHostAllocationReservation {
    pub fn reserve(bytes: u64, label: &str) -> Result<Self, RenderError> {
        let handle = tracked_host_allocation(bytes, label)?;
        Ok(Self {
            handle: Some(handle),
            bytes,
        })
    }

    pub fn release(&mut self) -> bool {
        self.handle.take().is_some()
    }

    pub fn is_active(&self) -> bool {
        self.handle.is_some()
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Quantize a screen coordinate to the deterministic integer grid.
pub fn quantize_coord(value: f32) -> i64 {
    (value as f64 * COORD_SCALE).round() as i64
}

/// Quantize a priority weight to the deterministic integer grid.
pub fn quantize_weight(weight: f64) -> i64 {
    (weight * WEIGHT_SCALE).round() as i64
}

/// Invalid candidate input. Candidate identity is the stable
/// `(label_id, candidate_index)` pair; duplicate identities are rejected
/// instead of being silently correlated after the solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateError {
    NonFiniteBounds { label_id: u64, candidate_index: u32 },
    NonFiniteWeight { label_id: u64, candidate_index: u32 },
    NegativeWeight { label_id: u64, candidate_index: u32 },
    OutOfRange { label_id: u64, candidate_index: u32 },
    DegenerateBounds { label_id: u64, candidate_index: u32 },
    DuplicateIdentity { label_id: u64, candidate_index: u32 },
    InvalidConfiguration { field: &'static str },
    ArithmeticOverflow { context: &'static str },
}

impl fmt::Display for CandidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (label_id, candidate_index, reason) = match self {
            Self::NonFiniteBounds {
                label_id,
                candidate_index,
            } => (*label_id, *candidate_index, "bounds must be finite"),
            Self::NonFiniteWeight {
                label_id,
                candidate_index,
            } => (*label_id, *candidate_index, "weight must be finite"),
            Self::NegativeWeight {
                label_id,
                candidate_index,
            } => (*label_id, *candidate_index, "weight must be non-negative"),
            Self::OutOfRange {
                label_id,
                candidate_index,
            } => (
                *label_id,
                *candidate_index,
                "quantized value is out of range",
            ),
            Self::DegenerateBounds {
                label_id,
                candidate_index,
            } => (
                *label_id,
                *candidate_index,
                "bounds must remain non-degenerate after quantization",
            ),
            Self::DuplicateIdentity {
                label_id,
                candidate_index,
            } => (
                *label_id,
                *candidate_index,
                "candidate identity is duplicated",
            ),
            Self::InvalidConfiguration { field } => {
                return write!(f, "invalid solver configuration: {field}")
            }
            Self::ArithmeticOverflow { context } => {
                return write!(f, "solver arithmetic overflow: {context}")
            }
        };
        write!(f, "candidate ({label_id}, {candidate_index}): {reason}")
    }
}

impl std::error::Error for CandidateError {}

/// A quantized candidate as seen by the optimal solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverCandidate {
    /// Label identifier.
    pub label_id: u64,
    /// Position index within the label's candidate set (total order key).
    pub candidate_index: u32,
    /// Quantized bounds `[min_x, min_y, max_x, max_y]` on the 1/16 px grid.
    pub bounds_q: [i64; 4],
    /// Quantized priority weight (1/1024 units).
    pub weight_q: i64,
    /// Caller-supplied eligibility gate. The bool itself carries no evidence
    /// about why a candidate was filtered.
    pub visible: bool,
}

impl SolverCandidate {
    /// Build a solver candidate from float bounds/weight, normalizing the
    /// box so `min <= max` on both axes.
    pub fn try_new(
        label_id: u64,
        candidate_index: u32,
        bounds: [f32; 4],
        weight: f64,
        visible: bool,
    ) -> Result<Self, CandidateError> {
        if !bounds.iter().all(|value| value.is_finite()) {
            return Err(CandidateError::NonFiniteBounds {
                label_id,
                candidate_index,
            });
        }
        if !weight.is_finite() {
            return Err(CandidateError::NonFiniteWeight {
                label_id,
                candidate_index,
            });
        }
        if weight < 0.0 {
            return Err(CandidateError::NegativeWeight {
                label_id,
                candidate_index,
            });
        }
        if bounds
            .iter()
            .map(|value| *value as f64 * COORD_SCALE)
            .chain(std::iter::once(weight * WEIGHT_SCALE))
            .any(|value| value < i64::MIN as f64 || value > i64::MAX as f64)
        {
            return Err(CandidateError::OutOfRange {
                label_id,
                candidate_index,
            });
        }
        let x0 = quantize_coord(bounds[0].min(bounds[2]));
        let y0 = quantize_coord(bounds[1].min(bounds[3]));
        let x1 = quantize_coord(bounds[0].max(bounds[2]));
        let y1 = quantize_coord(bounds[1].max(bounds[3]));
        if x0 >= x1 || y0 >= y1 {
            return Err(CandidateError::DegenerateBounds {
                label_id,
                candidate_index,
            });
        }
        Ok(Self {
            label_id,
            candidate_index,
            bounds_q: [x0, y0, x1, y1],
            weight_q: quantize_weight(weight),
            visible,
        })
    }

    /// Adapt an existing [`PlacementCandidate`] (the geometry authorities'
    /// output) into the solver's quantized model.
    pub fn try_from_placement(candidate: &PlacementCandidate) -> Result<Self, CandidateError> {
        Self::try_new(
            candidate.label_id,
            candidate.anchor_index,
            candidate.bounds,
            candidate.priority as f64,
            candidate.visible,
        )
    }

    fn total_order_key(&self) -> (u64, u32) {
        (self.label_id, self.candidate_index)
    }
}

/// Inclusive AABB intersection on the quantized grid (touching counts, to
/// match the compile-time Python `_rects_intersect` semantics).
fn boxes_conflict_q(a: &[i64; 4], b: &[i64; 4], margin_q: i64) -> bool {
    i128::from(a[0]) - i128::from(margin_q) <= i128::from(b[2])
        && i128::from(a[2]) + i128::from(margin_q) >= i128::from(b[0])
        && i128::from(a[1]) - i128::from(margin_q) <= i128::from(b[3])
        && i128::from(a[3]) + i128::from(margin_q) >= i128::from(b[1])
}

/// Overlap area between two quantized boxes, in quantized-square units.
fn overlap_area_q(a: &[i64; 4], b: &[i64; 4]) -> Result<i128, CandidateError> {
    let dx = (i128::from(a[2].min(b[2])) - i128::from(a[0].max(b[0]))).max(0);
    let dy = (i128::from(a[3].min(b[3])) - i128::from(a[1].max(b[1]))).max(0);
    dx.checked_mul(dy)
        .ok_or(CandidateError::ArithmeticOverflow {
            context: "overlap area",
        })
}

/// A typed, reproducible record of one solver decision.
#[derive(Debug, Clone, PartialEq)]
pub enum RationaleRecord {
    /// A candidate was placed; `displaced` lists conflicting candidates of
    /// other labels that were not placed, as `(label_id, candidate_index,
    /// overlap_area_q)`.
    Placed {
        label_id: u64,
        candidate_index: u32,
        weight_q: i64,
        displaced: Vec<(u64, u32, i128)>,
    },
    /// A label with visible candidates was not placed; `blocking` lists the
    /// placed candidates conflicting with its best candidate.
    Dropped {
        label_id: u64,
        candidate_index: u32,
        weight_q: i64,
        priority_lost: bool,
        blocking: Vec<(u64, u32, i128)>,
    },
    /// A candidate was excluded by the caller-supplied eligibility flag.
    /// Native bool input carries no depth/silhouette evidence.
    VisibilityFilteredCandidate { label_id: u64, candidate_index: u32 },
    /// Solve summary: node count, certification, and achieved gap.
    Solver {
        nodes_explored: u64,
        certified: bool,
        budget_exhausted: bool,
        objective_q: i128,
        upper_bound_q: i128,
        gap: f64,
        gap_tolerance: f64,
    },
}

/// Result of a bounded-optimal solve.
#[derive(Debug, Clone)]
pub struct OptimalOutcome {
    /// Chosen `(label_id, candidate_index)` pairs, sorted by label id.
    pub placements: Vec<(u64, u32)>,
    /// Achieved objective (sum of effective quantized weights).
    pub objective_q: i128,
    /// Certified upper bound on the optimum objective.
    pub upper_bound_q: i128,
    /// Certified optimality gap: `(upper_bound - objective) / upper_bound`.
    pub gap: f64,
    /// True when the returned objective is certified against
    /// `upper_bound_q`. Tolerance-pruned branches need not be enumerated:
    /// their recorded upper bounds are the proof. False means the work
    /// budget was hit; the gap remains honest but is not a tolerance
    /// certificate.
    pub certified: bool,
    /// Branch-and-bound nodes explored.
    pub nodes_explored: u64,
    /// Grounded decision records, in deterministic order.
    pub rationale: Vec<RationaleRecord>,
}

struct LabelGroup {
    label_id: u64,
    candidates: Vec<SolverCandidate>,
    max_weight: i128,
}

/// Capture the grounded records at the moment an incumbent is committed.
/// The record set is stored with that incumbent, so rationale identity and
/// conflicts cannot drift through a later post-hoc match by label alone.
fn record_committed_selection(
    groups: &[LabelGroup],
    selection: &[Option<usize>],
    ordered: &[SolverCandidate],
    margin_q: i64,
) -> Result<Vec<RationaleRecord>, CandidateError> {
    let placed_boxes: Vec<(u64, u32, [i64; 4], i64)> = groups
        .iter()
        .enumerate()
        .filter_map(|(group_pos, group)| {
            selection[group_pos].map(|pos| {
                let candidate = &group.candidates[pos];
                (
                    group.label_id,
                    candidate.candidate_index,
                    candidate.bounds_q,
                    candidate.weight_q,
                )
            })
        })
        .collect();
    let placed_keys: Vec<(u64, u32)> = placed_boxes
        .iter()
        .map(|(label_id, candidate_index, _, _)| (*label_id, *candidate_index))
        .collect();
    let mut records = Vec::new();
    let mut sorted_groups: Vec<(usize, &LabelGroup)> = groups.iter().enumerate().collect();
    sorted_groups.sort_by_key(|(_, group)| group.label_id);
    for (group_pos, group) in sorted_groups {
        match selection[group_pos] {
            Some(pos) => {
                let chosen = &group.candidates[pos];
                let displaced_candidates: Vec<&SolverCandidate> = ordered
                    .iter()
                    .filter(|other| {
                        other.visible
                            && other.label_id != group.label_id
                            && !placed_keys.contains(&(other.label_id, other.candidate_index))
                            && boxes_conflict_q(&chosen.bounds_q, &other.bounds_q, margin_q)
                    })
                    .collect();
                let mut displaced: Vec<(u64, u32, i128)> = Vec::new();
                for other in displaced_candidates {
                    displaced.push((
                        other.label_id,
                        other.candidate_index,
                        overlap_area_q(&chosen.bounds_q, &other.bounds_q)?,
                    ));
                }
                displaced.sort_unstable();
                records.push(RationaleRecord::Placed {
                    label_id: group.label_id,
                    candidate_index: chosen.candidate_index,
                    weight_q: chosen.weight_q,
                    displaced,
                });
            }
            None => {
                let best = group
                    .candidates
                    .iter()
                    .max_by(|a, b| {
                        a.weight_q
                            .cmp(&b.weight_q)
                            .then(b.candidate_index.cmp(&a.candidate_index))
                    })
                    .expect("group is non-empty");
                let blocking_candidates: Vec<&(u64, u32, [i64; 4], i64)> = placed_boxes
                    .iter()
                    .filter(|(_, _, bounds_q, _)| {
                        boxes_conflict_q(&best.bounds_q, bounds_q, margin_q)
                    })
                    .collect();
                let mut blocking: Vec<(u64, u32, i128)> = Vec::new();
                for (label_id, candidate_index, bounds_q, _) in blocking_candidates {
                    blocking.push((
                        *label_id,
                        *candidate_index,
                        overlap_area_q(&best.bounds_q, bounds_q)?,
                    ));
                }
                blocking.sort_unstable();
                let priority_lost = placed_boxes.iter().any(|(_, _, bounds_q, weight_q)| {
                    boxes_conflict_q(&best.bounds_q, bounds_q, margin_q)
                        && *weight_q > best.weight_q
                });
                records.push(RationaleRecord::Dropped {
                    label_id: group.label_id,
                    candidate_index: best.candidate_index,
                    weight_q: best.weight_q,
                    priority_lost,
                    blocking,
                });
            }
        }
    }
    Ok(records)
}

/// Bounded-optimal branch-and-bound solve over the candidate set.
pub fn declutter_optimal(
    candidates: &[SolverCandidate],
    config: &DeclutterConfig,
) -> Result<OptimalOutcome, CandidateError> {
    if !config.gap_tolerance.is_finite() || !(0.0..=1.0).contains(&config.gap_tolerance) {
        return Err(CandidateError::InvalidConfiguration {
            field: "gap_tolerance must be finite and within [0, 1]",
        });
    }
    if !config.margin.is_finite() || config.margin < 0.0 {
        return Err(CandidateError::InvalidConfiguration {
            field: "margin must be finite and non-negative",
        });
    }
    let margin_scaled = f64::from(config.margin) * COORD_SCALE;
    if margin_scaled > i64::MAX as f64 {
        return Err(CandidateError::InvalidConfiguration {
            field: "margin exceeds the quantized coordinate range",
        });
    }
    // Deterministic total order over all candidates.
    let mut ordered: Vec<SolverCandidate> = candidates.to_vec();
    ordered.sort_by_key(SolverCandidate::total_order_key);
    if let Some(pair) = ordered
        .windows(2)
        .find(|pair| pair[0].total_order_key() == pair[1].total_order_key())
    {
        return Err(CandidateError::DuplicateIdentity {
            label_id: pair[0].label_id,
            candidate_index: pair[0].candidate_index,
        });
    }

    let mut rationale: Vec<RationaleRecord> = Vec::new();
    for candidate in ordered.iter().filter(|candidate| !candidate.visible) {
        rationale.push(RationaleRecord::VisibilityFilteredCandidate {
            label_id: candidate.label_id,
            candidate_index: candidate.candidate_index,
        });
    }

    // Group visible candidates per label (BTreeMap: deterministic order).
    let mut grouped: BTreeMap<u64, Vec<SolverCandidate>> = BTreeMap::new();
    for candidate in ordered.iter().filter(|candidate| candidate.visible) {
        grouped
            .entry(candidate.label_id)
            .or_default()
            .push(candidate.clone());
    }

    let mut groups: Vec<LabelGroup> = grouped
        .into_iter()
        .map(|(label_id, candidates)| {
            let max_weight = candidates
                .iter()
                .map(|candidate| i128::from(candidate.weight_q))
                .max()
                .unwrap_or(0);
            LabelGroup {
                label_id,
                candidates,
                max_weight,
            }
        })
        .collect();
    // Branch order: strongest label first, then label id (deterministic).
    groups.sort_by(|a, b| {
        b.max_weight
            .cmp(&a.max_weight)
            .then(a.label_id.cmp(&b.label_id))
    });

    let n = groups.len();
    if n == 0 {
        rationale.push(RationaleRecord::Solver {
            nodes_explored: 0,
            certified: true,
            budget_exhausted: false,
            objective_q: 0,
            upper_bound_q: 0,
            gap: 0.0,
            gap_tolerance: config.gap_tolerance,
        });
        return Ok(OptimalOutcome {
            placements: Vec::new(),
            objective_q: 0,
            upper_bound_q: 0,
            gap: 0.0,
            certified: true,
            nodes_explored: 0,
            rationale,
        });
    }

    // suffix_max[k] = best possible remaining contribution from labels k..n.
    let mut suffix_max = vec![0i128; n + 1];
    for k in (0..n).rev() {
        suffix_max[k] = suffix_max[k + 1].checked_add(groups[k].max_weight).ok_or(
            CandidateError::ArithmeticOverflow {
                context: "objective upper bound",
            },
        )?;
    }
    let root_bound = suffix_max[0];
    // Floor keeps the fixed-point pruning threshold conservative: tolerance
    // quantization can never certify a gap larger than the caller requested.
    let tolerance_q = (config.gap_tolerance.clamp(0.0, 1.0) * GAP_SCALE as f64).floor() as i64;
    let margin_q = quantize_coord(config.margin.max(0.0));

    // Greedy incumbent: weight desc, then (label_id, candidate_index).
    let mut greedy_order: Vec<(usize, usize)> = Vec::new();
    for (group_pos, group) in groups.iter().enumerate() {
        for candidate_pos in 0..group.candidates.len() {
            greedy_order.push((group_pos, candidate_pos));
        }
    }
    greedy_order.sort_by(|&(ga, ca), &(gb, cb)| {
        let a = &groups[ga].candidates[ca];
        let b = &groups[gb].candidates[cb];
        b.weight_q
            .cmp(&a.weight_q)
            .then(a.total_order_key().cmp(&b.total_order_key()))
    });
    let mut best_selection: Vec<Option<usize>> = vec![None; n];
    let mut best_objective: i128 = 0;
    let mut best_cardinality: usize = 0;
    {
        let mut placed_boxes: Vec<[i64; 4]> = Vec::new();
        for (group_pos, candidate_pos) in greedy_order {
            if best_selection[group_pos].is_some() {
                continue;
            }
            let candidate = &groups[group_pos].candidates[candidate_pos];
            if placed_boxes
                .iter()
                .any(|placed| boxes_conflict_q(&candidate.bounds_q, placed, margin_q))
            {
                continue;
            }
            best_selection[group_pos] = Some(candidate_pos);
            best_cardinality =
                best_cardinality
                    .checked_add(1)
                    .ok_or(CandidateError::ArithmeticOverflow {
                        context: "greedy incumbent cardinality",
                    })?;
            best_objective = best_objective
                .checked_add(i128::from(candidate.weight_q))
                .ok_or(CandidateError::ArithmeticOverflow {
                    context: "greedy incumbent objective",
                })?;
            placed_boxes.push(candidate.bounds_q);
        }
    }
    let mut best_decisions =
        record_committed_selection(&groups, &best_selection, &ordered, margin_q)?;

    // Depth-first branch-and-bound with an explicit stack (no recursion, no
    // RNG, no wall clock). Frame cursor c: 0..len = candidate index, len =
    // skip, len+1 = exhausted.
    struct Frame {
        cursor: usize,
        chosen: Option<usize>,
    }
    let mut stack: Vec<Frame> = vec![Frame {
        cursor: 0,
        chosen: None,
    }];
    let mut committed: i128 = 0;
    let mut committed_cardinality: usize = 0;
    let mut current: Vec<Option<usize>> = vec![None; n];
    let mut nodes_explored: u64 = 0;
    let mut max_pruned_bound: i128 = 0;
    let mut budget_exceeded = false;

    while !stack.is_empty() {
        let depth = stack.len() - 1;
        // Undo the previous choice at this frame, if any.
        if let Some(prev) = stack[depth].chosen.take() {
            committed = committed
                .checked_sub(i128::from(groups[depth].candidates[prev].weight_q))
                .ok_or(CandidateError::ArithmeticOverflow {
                    context: "search backtrack objective",
                })?;
            committed_cardinality =
                committed_cardinality
                    .checked_sub(1)
                    .ok_or(CandidateError::ArithmeticOverflow {
                        context: "search backtrack cardinality",
                    })?;
            current[depth] = None;
        }
        let options = groups[depth].candidates.len();
        if stack[depth].cursor > options {
            stack.pop();
            continue;
        }
        if config
            .node_budget
            .is_some_and(|node_budget| nodes_explored >= node_budget)
        {
            budget_exceeded = true;
            break;
        }
        let cursor = stack[depth].cursor;
        stack[depth].cursor += 1;
        nodes_explored += 1;

        let (gain, gain_cardinality, feasible) = if cursor < options {
            let candidate = &groups[depth].candidates[cursor];
            let conflict = current[..depth]
                .iter()
                .enumerate()
                .any(|(prior_depth, chosen)| match chosen {
                    Some(pos) => boxes_conflict_q(
                        &candidate.bounds_q,
                        &groups[prior_depth].candidates[*pos].bounds_q,
                        margin_q,
                    ),
                    None => false,
                });
            (i128::from(candidate.weight_q), 1usize, !conflict)
        } else {
            (0, 0, true) // skip
        };
        if !feasible {
            continue;
        }
        let bound = committed
            .checked_add(gain)
            .and_then(|value| value.checked_add(suffix_max[depth + 1]))
            .ok_or(CandidateError::ArithmeticOverflow {
                context: "branch upper bound",
            })?;
        let cardinality_bound = committed_cardinality
            .checked_add(gain_cardinality)
            .and_then(|value| value.checked_add(n - depth - 1))
            .ok_or(CandidateError::ArithmeticOverflow {
                context: "cardinality upper bound",
            })?;
        let tolerance_prunable = bound > best_objective
            && bound
                .checked_sub(best_objective)
                .and_then(|difference| difference.checked_mul(i128::from(GAP_SCALE)))
                .zip(bound.checked_mul(i128::from(tolerance_q)))
                .map(|(difference, tolerance)| difference <= tolerance)
                .ok_or(CandidateError::ArithmeticOverflow {
                    context: "relative gap comparison",
                })?;
        let within_tolerance = bound < best_objective
            || (bound == best_objective && cardinality_bound <= best_cardinality)
            || tolerance_prunable;
        if within_tolerance {
            max_pruned_bound = max_pruned_bound.max(bound);
            continue;
        }
        // Commit this option.
        if cursor < options {
            committed = committed
                .checked_add(gain)
                .ok_or(CandidateError::ArithmeticOverflow {
                    context: "search objective",
                })?;
            committed_cardinality =
                committed_cardinality
                    .checked_add(1)
                    .ok_or(CandidateError::ArithmeticOverflow {
                        context: "search cardinality",
                    })?;
            current[depth] = Some(cursor);
            stack[depth].chosen = Some(cursor);
        }
        if depth + 1 == n {
            if committed > best_objective
                || (committed == best_objective && committed_cardinality > best_cardinality)
            {
                best_objective = committed;
                best_cardinality = committed_cardinality;
                best_selection.copy_from_slice(&current);
                best_decisions =
                    record_committed_selection(&groups, &best_selection, &ordered, margin_q)?;
            }
        } else {
            stack.push(Frame {
                cursor: 0,
                chosen: None,
            });
        }
    }

    let certified = !budget_exceeded;
    let final_ub = if certified {
        best_objective.max(max_pruned_bound)
    } else {
        root_bound
    };
    let gap = if final_ub <= 0 {
        0.0
    } else {
        (final_ub - best_objective) as f64 / final_ub as f64
    };

    // Materialize placements sorted by label id.
    let mut placements: Vec<(u64, u32)> = groups
        .iter()
        .enumerate()
        .filter_map(|(group_pos, group)| {
            best_selection[group_pos]
                .map(|pos| (group.label_id, group.candidates[pos].candidate_index))
        })
        .collect();
    placements.sort_unstable();

    rationale.extend(best_decisions);
    rationale.push(RationaleRecord::Solver {
        nodes_explored,
        certified,
        budget_exhausted: budget_exceeded,
        objective_q: best_objective,
        upper_bound_q: final_ub,
        gap,
        gap_tolerance: config.gap_tolerance,
    });

    Ok(OptimalOutcome {
        placements,
        objective_q: best_objective,
        upper_bound_q: final_ub,
        gap,
        certified,
        nodes_explored,
        rationale,
    })
}

/// Adapter for the [`super::declutter::DeclutterAlgorithm::Optimal`] arm:
/// runs the bounded-optimal solve over [`PlacementCandidate`]s and shapes
/// the answer as a [`DeclutterResult`].
pub fn declutter_optimal_result(
    candidates: Vec<PlacementCandidate>,
    config: &DeclutterConfig,
) -> Result<DeclutterResult, CandidateError> {
    let solver_candidates: Vec<SolverCandidate> = candidates
        .iter()
        .map(SolverCandidate::try_from_placement)
        .collect::<Result<_, _>>()?;
    let outcome = declutter_optimal(&solver_candidates, config)?;
    let mut visible_labels = Vec::with_capacity(outcome.placements.len());
    let mut positions = Vec::with_capacity(outcome.placements.len());
    for &(label_id, candidate_index) in &outcome.placements {
        visible_labels.push(label_id);
        if let Some(candidate) = candidates
            .iter()
            .find(|c| c.label_id == label_id && c.anchor_index == candidate_index)
        {
            positions.push((label_id, candidate.position));
        }
    }
    Ok(DeclutterResult {
        visible_labels,
        positions,
        total_energy: -(outcome.objective_q as f32) / WEIGHT_SCALE as f32,
        iterations: outcome.nodes_explored as usize,
    })
}

/// The 8-position cartographic ladder around an anchor, in preference
/// order: NE, NW, SE, SW, E, W, N, S (`anchor_index` 0..=7).
pub fn ladder_candidates(
    label_id: u64,
    anchor: [f32; 2],
    half_extent: [f32; 2],
    offset: f32,
    priority: i32,
) -> Vec<PlacementCandidate> {
    const DIRECTIONS: [[f32; 2]; 8] = [
        [1.0, -1.0],  // NE (screen y grows downward)
        [-1.0, -1.0], // NW
        [1.0, 1.0],   // SE
        [-1.0, 1.0],  // SW
        [1.0, 0.0],   // E
        [-1.0, 0.0],  // W
        [0.0, -1.0],  // N
        [0.0, 1.0],   // S
    ];
    DIRECTIONS
        .iter()
        .enumerate()
        .map(|(index, direction)| {
            let position = [
                anchor[0] + direction[0] * offset,
                anchor[1] + direction[1] * offset,
            ];
            PlacementCandidate {
                label_id,
                anchor_index: index as u32,
                position,
                bounds: [
                    position[0] - half_extent[0],
                    position[1] - half_extent[1],
                    position[0] + half_extent[0],
                    position[1] + half_extent[1],
                ],
                priority,
                cost: index as f32 * 0.001,
                selected: false,
                visible: true,
            }
        })
        .collect()
}

/// Exact screen AABB of a rendered glyph quad. Atlas dimensions, bearings,
/// shaping/GPOS offsets, projection, and scale are already baked into the
/// [`TextInstance`] rectangle; this adapter only applies its recorded rotation.
fn rendered_quad_aabb(instance: &TextInstance) -> Option<[f32; 4]> {
    if !instance.rect_min.iter().all(|value| value.is_finite())
        || !instance.rect_max.iter().all(|value| value.is_finite())
        || !instance.rotation.is_finite()
        || instance.rect_min[0] >= instance.rect_max[0]
        || instance.rect_min[1] >= instance.rect_max[1]
    {
        return None;
    }
    let min_x = f64::from(instance.rect_min[0]);
    let min_y = f64::from(instance.rect_min[1]);
    let max_x = f64::from(instance.rect_max[0]);
    let max_y = f64::from(instance.rect_max[1]);
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let half_width = (max_x - min_x) * 0.5;
    let half_height = (max_y - min_y) * 0.5;
    let (sin, cos) = f64::from(instance.rotation).sin_cos();
    let extent_x = cos.abs() * half_width + sin.abs() * half_height;
    let extent_y = sin.abs() * half_width + cos.abs() * half_height;
    let bounds = [
        center_x - extent_x,
        center_y - extent_y,
        center_x + extent_x,
        center_y + extent_y,
    ];
    if bounds
        .iter()
        .any(|value| !value.is_finite() || value.abs() > f64::from(f32::MAX))
    {
        return None;
    }
    Some([
        bounds[0] as f32,
        bounds[1] as f32,
        bounds[2] as f32,
        bounds[3] as f32,
    ])
}

/// Build a candidate by unioning the actual GPU glyph quads emitted by the
/// atlas/shaping authority. No glyph layout is recomputed here.
pub fn candidate_from_rendered_glyphs(
    label_id: u64,
    anchor_index: u32,
    instances: &[TextInstance],
    priority: i32,
) -> Option<PlacementCandidate> {
    if instances.is_empty() {
        return None;
    }
    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for instance in instances {
        let glyph_bounds = rendered_quad_aabb(instance)?;
        bounds[0] = bounds[0].min(glyph_bounds[0]);
        bounds[1] = bounds[1].min(glyph_bounds[1]);
        bounds[2] = bounds[2].max(glyph_bounds[2]);
        bounds[3] = bounds[3].max(glyph_bounds[3]);
    }
    let position = [(bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5];
    Some(PlacementCandidate {
        label_id,
        anchor_index,
        position,
        bounds,
        priority,
        cost: 0.0,
        selected: false,
        visible: true,
    })
}

/// Line-label adapter. The input must be the rendered quads returned by
/// `MsdfAtlas::layout_shaped_on_placements`.
pub fn candidate_from_line_instances(
    label_id: u64,
    anchor_index: u32,
    instances: &[TextInstance],
    priority: i32,
) -> Option<PlacementCandidate> {
    candidate_from_rendered_glyphs(label_id, anchor_index, instances, priority)
}

/// Curved-label adapter. Curved projection and atlas layout must happen once
/// upstream; the same authoritative rendered-quad union is then used here.
pub fn candidate_from_curved_instances(
    label_id: u64,
    anchor_index: u32,
    instances: &[TextInstance],
    priority: i32,
) -> Option<PlacementCandidate> {
    candidate_from_rendered_glyphs(label_id, anchor_index, instances, priority)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static DEPTH_HOST_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn cand(label: u64, index: u32, bounds: [f32; 4], weight: f64) -> SolverCandidate {
        SolverCandidate::try_new(label, index, bounds, weight, true)
            .expect("test candidate must be valid")
    }

    fn config() -> DeclutterConfig {
        DeclutterConfig {
            margin: 0.0,
            ..DeclutterConfig::default()
        }
    }

    fn solve(candidates: &[SolverCandidate], config: &DeclutterConfig) -> OptimalOutcome {
        declutter_optimal(candidates, config).expect("valid unique test candidates")
    }

    /// Independent exhaustive oracle over the per-label choice product. It
    /// sums the raw quantized priorities directly rather than sharing solver
    /// objective helpers.
    fn brute_force_optimum(candidates: &[SolverCandidate], margin_q: i64) -> i128 {
        let mut grouped: BTreeMap<u64, Vec<&SolverCandidate>> = BTreeMap::new();
        for candidate in candidates.iter().filter(|candidate| candidate.visible) {
            grouped
                .entry(candidate.label_id)
                .or_default()
                .push(candidate);
        }
        let groups: Vec<Vec<&SolverCandidate>> = grouped.into_values().collect();
        let mut best = 0i128;
        let mut choice = vec![0usize; groups.len()];
        loop {
            // Evaluate: index == len(group) means skip.
            let mut objective = 0i128;
            let mut boxes: Vec<[i64; 4]> = Vec::new();
            let mut feasible = true;
            for (group, &pick) in groups.iter().zip(choice.iter()) {
                if pick == group.len() {
                    continue;
                }
                let candidate = group[pick];
                if boxes
                    .iter()
                    .any(|other| boxes_conflict_q(&candidate.bounds_q, other, margin_q))
                {
                    feasible = false;
                    break;
                }
                boxes.push(candidate.bounds_q);
                objective += i128::from(candidate.weight_q);
            }
            if feasible {
                best = best.max(objective);
            }
            // Advance mixed-radix counter.
            let mut done = true;
            for (slot, group) in choice.iter_mut().zip(groups.iter()) {
                if *slot < group.len() {
                    *slot += 1;
                    done = false;
                    break;
                }
                *slot = 0;
            }
            if done {
                break;
            }
        }
        best
    }

    fn assert_within_gap(candidates: &[SolverCandidate]) {
        let cfg = config();
        let outcome = solve(candidates, &cfg);
        let optimum = brute_force_optimum(candidates, 0);
        assert!(
            outcome.objective_q as f64 >= 0.98 * optimum as f64,
            "objective {} below 98% of brute-force optimum {}",
            outcome.objective_q,
            optimum
        );
        assert!(
            outcome.gap <= cfg.gap_tolerance + 1e-12,
            "reported gap {} exceeds tolerance",
            outcome.gap
        );
        assert!(outcome.certified, "small instance must certify");
    }

    #[test]
    fn test_optimal_beats_greedy_on_chain() {
        // Chain A-B-C: B overlaps both, w(B)=10 > w(A)=6, w(C)=6.
        // Greedy places B (10); optimal places A+C (12).
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 10.0, 10.0], 6.0),
            cand(2, 0, [5.0, 0.0, 15.0, 10.0], 10.0),
            cand(3, 0, [12.0, 0.0, 22.0, 10.0], 6.0),
        ];
        let outcome = solve(&candidates, &config());
        let placed: Vec<u64> = outcome.placements.iter().map(|(id, _)| *id).collect();
        assert_eq!(placed, vec![1, 3]);
        assert_within_gap(&candidates);
    }

    #[test]
    fn test_raw_priority_beats_cardinality_bonus_reviewer_repro() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 30.0, 10.0], 10.0 / WEIGHT_SCALE),
            cand(2, 0, [0.0, 0.0, 9.0, 10.0], 3.0 / WEIGHT_SCALE),
            cand(3, 0, [10.5, 0.0, 19.5, 10.0], 3.0 / WEIGHT_SCALE),
            cand(4, 0, [21.0, 0.0, 30.0, 10.0], 3.0 / WEIGHT_SCALE),
        ];
        let outcome = solve(&candidates, &config());
        assert_eq!(outcome.placements, vec![(1, 0)]);
        assert_eq!(outcome.objective_q, 10);
        assert_eq!(outcome.upper_bound_q, 10);
        assert_eq!(outcome.gap, 0.0);
        assert!(outcome.certified);
        assert_eq!(brute_force_optimum(&candidates, 0), 10);
    }

    #[test]
    fn test_optimal_matches_bruteforce_on_hand_instances() {
        let instances: Vec<Vec<SolverCandidate>> = vec![
            // Non-overlapping: place everything.
            vec![
                cand(1, 0, [0.0, 0.0, 5.0, 5.0], 1.0),
                cand(2, 0, [10.0, 0.0, 15.0, 5.0], 2.0),
                cand(3, 0, [20.0, 0.0, 25.0, 5.0], 3.0),
            ],
            // Same-box pair: keep the heavier one.
            vec![
                cand(1, 0, [50.0, 50.0, 51.0, 51.0], 5.0),
                cand(2, 0, [50.0, 50.0, 51.0, 51.0], 9.0),
            ],
            // Two candidates per label with cross conflicts.
            vec![
                cand(1, 0, [0.0, 0.0, 10.0, 10.0], 8.0),
                cand(1, 1, [20.0, 0.0, 30.0, 10.0], 7.0),
                cand(2, 0, [5.0, 5.0, 15.0, 15.0], 6.0),
                cand(2, 1, [40.0, 0.0, 50.0, 10.0], 5.0),
                cand(3, 0, [25.0, 5.0, 35.0, 15.0], 4.0),
            ],
            // Star: center conflicts with four satellites.
            vec![
                cand(1, 0, [10.0, 10.0, 30.0, 30.0], 11.0),
                cand(2, 0, [5.0, 5.0, 15.0, 15.0], 4.0),
                cand(3, 0, [25.0, 5.0, 35.0, 15.0], 4.0),
                cand(4, 0, [5.0, 25.0, 15.0, 35.0], 4.0),
                cand(5, 0, [25.0, 25.0, 35.0, 35.0], 4.0),
            ],
            // Zero-weight labels still get placed when they fit.
            vec![
                cand(1, 0, [0.0, 0.0, 5.0, 5.0], 0.0),
                cand(2, 0, [10.0, 0.0, 15.0, 5.0], 0.0),
            ],
            // Fractional weights on the quantized grid.
            vec![
                cand(1, 0, [0.0, 0.0, 8.0, 8.0], 1.5),
                cand(2, 0, [4.0, 4.0, 12.0, 12.0], 1.25),
                cand(3, 0, [9.0, 0.0, 17.0, 7.0], 1.75),
            ],
        ];
        for candidates in &instances {
            assert_within_gap(candidates);
        }
    }

    #[test]
    fn test_deterministic_across_repeated_calls() {
        let candidates = vec![
            cand(3, 1, [0.0, 0.0, 10.0, 10.0], 5.0),
            cand(1, 0, [5.0, 5.0, 15.0, 15.0], 5.0),
            cand(2, 0, [8.0, 0.0, 18.0, 10.0], 5.0),
            cand(3, 0, [2.0, 2.0, 12.0, 12.0], 5.0),
        ];
        let cfg = config();
        let first = solve(&candidates, &cfg);
        for _ in 0..5 {
            let again = solve(&candidates, &cfg);
            assert_eq!(again.placements, first.placements);
            assert_eq!(again.objective_q, first.objective_q);
            assert_eq!(again.rationale, first.rationale);
            assert_eq!(again.nodes_explored, first.nodes_explored);
        }

        let mut reversed = candidates.clone();
        reversed.reverse();
        let reordered = solve(&reversed, &cfg);
        assert_eq!(reordered.placements, first.placements);
        assert_eq!(reordered.rationale, first.rationale);
    }

    #[test]
    fn test_invalid_and_duplicate_candidates_are_rejected() {
        assert!(matches!(
            SolverCandidate::try_new(1, 0, [0.0, f32::NAN, 1.0, 1.0], 1.0, true),
            Err(CandidateError::NonFiniteBounds { .. })
        ));
        assert!(matches!(
            SolverCandidate::try_new(1, 0, [0.0, 0.0, 1.0, 1.0], f64::INFINITY, true),
            Err(CandidateError::NonFiniteWeight { .. })
        ));
        assert!(matches!(
            SolverCandidate::try_new(1, 0, [0.0, 0.0, 1.0, 1.0], -1.0, true),
            Err(CandidateError::NegativeWeight { .. })
        ));
        assert!(matches!(
            SolverCandidate::try_new(1, 0, [0.0, 0.0, 0.0, 1.0], 1.0, true),
            Err(CandidateError::DegenerateBounds { .. })
        ));
        assert!(matches!(
            SolverCandidate::try_new(1, 0, [0.0, 0.0, 0.01, 1.0], 1.0, true),
            Err(CandidateError::DegenerateBounds { .. })
        ));
        let duplicate = cand(7, 3, [0.0, 0.0, 1.0, 1.0], 1.0);
        assert!(matches!(
            declutter_optimal(&[duplicate.clone(), duplicate], &config()),
            Err(CandidateError::DuplicateIdentity {
                label_id: 7,
                candidate_index: 3,
            })
        ));
    }

    #[test]
    fn test_candidate_identity_selects_at_most_one_per_label() {
        let candidates = vec![
            cand(4, 9, [0.0, 0.0, 4.0, 4.0], 3.0),
            cand(4, 2, [10.0, 0.0, 14.0, 4.0], 3.0),
        ];
        let outcome = solve(&candidates, &config());
        assert_eq!(outcome.placements, vec![(4, 2)]);
    }

    #[test]
    fn test_visibility_filtered_candidates_never_chosen_or_overclaimed() {
        let candidates = vec![
            SolverCandidate::try_new(1, 0, [0.0, 0.0, 10.0, 10.0], 100.0, false)
                .expect("valid candidate"),
            SolverCandidate::try_new(1, 1, [20.0, 0.0, 30.0, 10.0], 1.0, true)
                .expect("valid candidate"),
            SolverCandidate::try_new(2, 0, [50.0, 0.0, 60.0, 10.0], 50.0, false)
                .expect("valid candidate"),
        ];
        let outcome = solve(&candidates, &config());
        assert_eq!(outcome.placements, vec![(1, 1)]);
        let filtered: Vec<(u64, u32)> = outcome
            .rationale
            .iter()
            .filter_map(|record| match record {
                RationaleRecord::VisibilityFilteredCandidate {
                    label_id,
                    candidate_index,
                } => Some((*label_id, *candidate_index)),
                _ => None,
            })
            .collect();
        assert_eq!(filtered, vec![(1, 0), (2, 0)]);
    }

    #[test]
    fn test_objective_aggregate_exceeding_i64_is_supported() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 1.0, 1.0], 5.0e15),
            cand(2, 0, [2.0, 0.0, 3.0, 1.0], 5.0e15),
            cand(3, 0, [4.0, 0.0, 5.0, 1.0], 5.0e15),
        ];
        let outcome = solve(&candidates, &config());
        assert_eq!(outcome.placements, vec![(1, 0), (2, 0), (3, 0)]);
        assert!(outcome.objective_q > i128::from(i64::MAX));
        assert_eq!(outcome.objective_q, outcome.upper_bound_q);
        assert!(outcome.certified);
    }

    #[test]
    fn test_large_box_overlap_area_exceeding_i64_is_supported() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 1.0e9, 1.0e9], 2.0),
            cand(2, 0, [0.0, 0.0, 1.0e9, 1.0e9], 1.0),
        ];
        let outcome = solve(&candidates, &config());
        let area = outcome.rationale.iter().find_map(|record| match record {
            RationaleRecord::Placed { displaced, .. } => displaced.first().map(|entry| entry.2),
            _ => None,
        });
        assert!(area.expect("conflict area is recorded") > i128::from(i64::MAX));
    }

    #[test]
    fn test_budget_exceeded_returns_honest_gap() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 10.0, 10.0], 6.0),
            cand(2, 0, [5.0, 0.0, 15.0, 10.0], 10.0),
            cand(3, 0, [12.0, 0.0, 22.0, 10.0], 6.0),
        ];
        let cfg = DeclutterConfig {
            margin: 0.0,
            node_budget: Some(1),
            ..DeclutterConfig::default()
        };
        let outcome = solve(&candidates, &cfg);
        assert!(!outcome.certified, "budget-hit solve must not certify");
        assert!(outcome.gap > 0.0, "budget-hit gap must be honest (> 0)");
        // Incumbent is at least the greedy solution.
        assert!(!outcome.placements.is_empty());
    }

    #[test]
    fn test_rationale_records_ground_the_solution() {
        let candidates = vec![
            cand(1, 0, [0.0, 0.0, 10.0, 10.0], 9.0),
            cand(2, 0, [5.0, 0.0, 15.0, 10.0], 2.0),
        ];
        let outcome = solve(&candidates, &config());
        assert_eq!(outcome.placements, vec![(1, 0)]);
        let placed = outcome.rationale.iter().find_map(|record| match record {
            RationaleRecord::Placed {
                label_id,
                displaced,
                ..
            } if *label_id == 1 => Some(displaced.clone()),
            _ => None,
        });
        // Placed label 1 displaced label 2's candidate; overlap 5x10 px =
        // (5*16)*(10*16) quantized units.
        assert_eq!(placed, Some(vec![(2u64, 0u32, 80i128 * 160i128)]));
        let dropped = outcome.rationale.iter().find_map(|record| match record {
            RationaleRecord::Dropped {
                label_id,
                priority_lost,
                blocking,
                ..
            } if *label_id == 2 => Some((*priority_lost, blocking.clone())),
            _ => None,
        });
        let (priority_lost, blocking) = dropped.expect("label 2 dropped");
        assert!(priority_lost);
        assert_eq!(blocking, vec![(1u64, 0u32, 80i128 * 160i128)]);
    }

    #[test]
    fn test_ladder_candidates_shape() {
        let ladder = ladder_candidates(7, [100.0, 100.0], [20.0, 8.0], 12.0, 3);
        assert_eq!(ladder.len(), 8);
        for (index, candidate) in ladder.iter().enumerate() {
            assert_eq!(candidate.label_id, 7);
            assert_eq!(candidate.anchor_index, index as u32);
            assert!(candidate.visible);
            assert!((candidate.bounds[2] - candidate.bounds[0] - 40.0).abs() < 1e-5);
            assert!((candidate.bounds[3] - candidate.bounds[1] - 16.0).abs() < 1e-5);
        }
        // NE candidate sits up-right of the anchor.
        assert!(ladder[0].position[0] > 100.0 && ladder[0].position[1] < 100.0);
    }

    fn rendered_instance(rect: [f32; 4], rotation: f32) -> TextInstance {
        let mut instance = TextInstance::new(
            [rect[0], rect[1]],
            [rect[2], rect[3]],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0; 4],
        );
        instance.rotation = rotation;
        instance
    }

    #[test]
    fn test_line_candidate_unions_authoritative_rotated_non_square_quads() {
        let instances = vec![
            rendered_instance([4.0, 14.0, 16.0, 26.0], 0.0),
            rendered_instance([36.0, 18.0, 44.0, 34.0], std::f32::consts::FRAC_PI_4),
        ];
        let candidate = candidate_from_line_instances(9, 0, &instances, 5).expect("candidate");
        let rotated_extent = 6.0 * std::f32::consts::SQRT_2;
        let expected = [4.0, 14.0, 40.0 + rotated_extent, 26.0 + rotated_extent];
        for (actual, expected) in candidate.bounds.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert!((candidate.position[0] - (4.0 + 40.0 + rotated_extent) * 0.5).abs() < 1.0e-5);
        assert!((candidate.position[1] - (14.0 + 26.0 + rotated_extent) * 0.5).abs() < 1.0e-5);
        assert!(candidate_from_line_instances(9, 0, &[], 5).is_none());
    }

    #[test]
    fn test_line_adapter_consumes_nonidentity_projected_authority_output() {
        let view_proj = glam::Mat4::from_scale(glam::Vec3::new(0.01, 0.02, 1.0));
        let placements = super::super::line_label::compute_line_label_placement(
            &[
                glam::Vec3::new(-50.0, -10.0, 0.0),
                glam::Vec3::new(50.0, 10.0, 0.0),
            ],
            "AB",
            &[10.0, 10.0],
            view_proj,
            200.0,
            100.0,
            super::super::types::LineLabelPlacement::Along,
            14.0,
        );
        assert_eq!(placements.len(), 2);
        assert!(placements.iter().all(|glyph| glyph.scale == 14.0));
        assert!(placements.iter().all(|glyph| glyph.rotation.abs() > 0.0));
        let instances = vec![
            rendered_instance(
                [
                    placements[0].screen_pos[0] - 1.0,
                    placements[0].screen_pos[1] - 10.0,
                    placements[0].screen_pos[0] + 5.0,
                    placements[0].screen_pos[1] + 4.0,
                ],
                placements[0].rotation,
            ),
            rendered_instance(
                [
                    placements[1].screen_pos[0] - 6.5,
                    placements[1].screen_pos[1],
                    placements[1].screen_pos[0] + 4.5,
                    placements[1].screen_pos[1] + 8.0,
                ],
                placements[1].rotation,
            ),
        ];
        let candidate = candidate_from_line_instances(10, 0, &instances, 4)
            .expect("projected line glyph geometry yields a candidate");
        assert!(candidate.position[0] > 50.0 && candidate.position[0] < 150.0);
        assert!(candidate.position[1] > 25.0 && candidate.position[1] < 75.0);
        assert!(candidate.bounds[2] - candidate.bounds[0] > 14.0);
    }

    #[test]
    fn test_curved_adapter_uses_projected_rendered_quads_without_relayout() {
        let layout = super::super::curved::CurvedTextLayout {
            glyphs: vec![
                super::super::curved::CurvedGlyphInstance {
                    world_pos: glam::Vec3::new(-0.5, 0.0, 0.0),
                    rotation: 0.75,
                    uv_rect: [0.0, 0.0, 0.5, 0.5],
                    color: [1.0; 4],
                    scale: 12.0,
                    path_offset: 2.0,
                    character: 'A',
                },
                super::super::curved::CurvedGlyphInstance {
                    world_pos: glam::Vec3::new(0.5, 0.2, 0.0),
                    rotation: -0.25,
                    uv_rect: [0.5, 0.0, 1.0, 0.5],
                    color: [1.0; 4],
                    scale: 12.0,
                    path_offset: 8.0,
                    character: 'B',
                },
            ],
            total_width: 20.0,
            success: true,
        };
        let original = layout.glyphs.clone();
        let view_proj = glam::Mat4::from_scale(glam::Vec3::new(0.5, 0.25, 1.0));
        let projected =
            super::super::curved::project_curved_glyphs(&layout, view_proj, 200.0, 100.0);
        let instances = vec![
            rendered_instance(
                [
                    projected[0].screen_pos[0] - 3.0,
                    projected[0].screen_pos[1] - 8.0,
                    projected[0].screen_pos[0] + 7.0,
                    projected[0].screen_pos[1] + 4.0,
                ],
                projected[0].rotation,
            ),
            rendered_instance(
                [
                    projected[1].screen_pos[0] - 5.0,
                    projected[1].screen_pos[1] - 2.0,
                    projected[1].screen_pos[0] + 4.0,
                    projected[1].screen_pos[1] + 7.0,
                ],
                projected[1].rotation,
            ),
        ];
        let candidate = candidate_from_curved_instances(5, 4, &instances, 8)
            .expect("projected rendered glyph quads yield a candidate");
        let expected = candidate_from_rendered_glyphs(5, 4, &instances, 8)
            .expect("shared rendered-quad path yields a candidate");
        assert_eq!(candidate.bounds, expected.bounds);
        assert_eq!(candidate.position, expected.position);
        assert_ne!(candidate.position, [0.0, 0.1]);
        assert_ne!(projected[0].rotation, layout.glyphs[0].rotation);
        assert_eq!(layout.glyphs[0].rotation, original[0].rotation);
        assert_eq!(layout.glyphs[1].path_offset, original[1].path_offset);
    }

    #[test]
    fn test_depth_host_reservation_lifetime_tracks_and_releases() {
        let _test_guard = DEPTH_HOST_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tracker = crate::core::memory_tracker::global_tracker();
        let before = tracker.get_metrics().host_visible_bytes;
        {
            let reservation =
                DepthHostAllocationReservation::reserve(4096, "labels.depth.test.lifetime")
                    .expect("reservation fits budget");
            assert!(reservation.is_active());
            assert_eq!(reservation.bytes(), 4096);
            assert_eq!(tracker.get_metrics().host_visible_bytes, before + 4096);
        }
        assert_eq!(tracker.get_metrics().host_visible_bytes, before);
    }

    #[test]
    fn test_depth_host_reservation_release_is_exactly_once() {
        let _test_guard = DEPTH_HOST_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tracker = crate::core::memory_tracker::global_tracker();
        let before = tracker.get_metrics().host_visible_bytes;
        let mut reservation =
            DepthHostAllocationReservation::reserve(2048, "labels.depth.test.release")
                .expect("reservation fits budget");
        assert!(reservation.release());
        assert!(!reservation.release());
        assert!(!reservation.is_active());
        assert_eq!(tracker.get_metrics().host_visible_bytes, before);
        drop(reservation);
        assert_eq!(tracker.get_metrics().host_visible_bytes, before);
    }

    #[test]
    fn test_concurrent_depth_reservations_enforce_aggregate_budget() {
        let _test_guard = DEPTH_HOST_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tracker = crate::core::memory_tracker::global_tracker();
        assert_eq!(tracker.get_budget_policy(), "enforce");
        let before = tracker.get_metrics().host_visible_bytes;
        let available = tracker
            .get_budget_limit()
            .checked_sub(before)
            .expect("tracker is within its budget before test");
        let request = available / 2 + 1;
        let (reserved_tx, reserved_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let first = std::thread::spawn(move || {
            let reservation = DepthHostAllocationReservation::reserve(
                request,
                "labels.depth.test.concurrent.first",
            )
            .expect("first aggregate reservation fits");
            reserved_tx.send(()).expect("signal reservation");
            release_rx.recv().expect("release signal");
            drop(reservation);
        });
        reserved_rx.recv().expect("first reservation is active");
        assert_eq!(tracker.get_metrics().host_visible_bytes, before + request);
        let second =
            DepthHostAllocationReservation::reserve(request, "labels.depth.test.concurrent.second");
        assert!(matches!(second, Err(RenderError::Budget(_))));
        release_tx.send(()).expect("release first reservation");
        first.join().expect("reservation thread completes");
        assert_eq!(tracker.get_metrics().host_visible_bytes, before);
    }
}
