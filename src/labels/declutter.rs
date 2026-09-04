//! Annealing-based label decluttering.
//!
//! Uses simulated annealing or greedy algorithms to find optimal
//! label placements that minimize overlap and maximize readability.

use std::collections::HashSet;

/// A candidate label placement.
#[derive(Debug, Clone)]
pub struct PlacementCandidate {
    /// Label identifier.
    pub label_id: u64,
    /// Index of this candidate anchor within its label's candidate set.
    /// `(label_id, anchor_index)` is the deterministic total order used by
    /// the optimal solver for tie-breaking.
    pub anchor_index: u32,
    /// Screen position (x, y).
    pub position: [f32; 2],
    /// Bounding box: [min_x, min_y, max_x, max_y].
    pub bounds: [f32; 4],
    /// Priority (higher = more important).
    pub priority: i32,
    /// Cost of this placement (lower = better).
    pub cost: f32,
    /// Whether this candidate is currently selected.
    pub selected: bool,
    /// Caller-supplied eligibility gate. `false` candidates contribute no
    /// placements; the bool does not encode why they were filtered.
    pub visible: bool,
}

/// Declutter algorithm configuration.
#[derive(Debug, Clone)]
pub struct DeclutterConfig {
    /// Maximum iterations for annealing.
    pub max_iterations: usize,
    /// Initial temperature for annealing.
    pub initial_temperature: f32,
    /// Cooling rate (0.9-0.99 typical).
    pub cooling_rate: f32,
    /// Weight for overlap penalty in energy function.
    pub overlap_weight: f32,
    /// Weight for priority in energy function.
    pub priority_weight: f32,
    /// Weight for distance from preferred position.
    pub distance_weight: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Margin around labels for collision.
    pub margin: f32,
    /// Certified optimality gap tolerance for the optimal solver
    /// (fraction of the objective upper bound; 0.02 = 2%).
    pub gap_tolerance: f64,
    /// Optional deterministic branch-and-bound work budget. `None` means
    /// unbounded work. A work budget is used instead of a wall-clock limit
    /// because elapsed time is not a reproducible input to placement.
    pub node_budget: Option<u64>,
}

impl Default for DeclutterConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            overlap_weight: 10.0,
            priority_weight: 1.0,
            distance_weight: 0.5,
            seed: 42,
            margin: 2.0,
            gap_tolerance: 0.02,
            node_budget: None,
        }
    }
}

/// Result of decluttering operation.
#[derive(Debug, Clone)]
pub struct DeclutterResult {
    /// Selected label IDs that should be displayed.
    pub visible_labels: Vec<u64>,
    /// Final positions for visible labels.
    pub positions: Vec<(u64, [f32; 2])>,
    /// Total energy of the solution.
    pub total_energy: f32,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Simple pseudo-random number generator for reproducibility.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// Check if two bounding boxes overlap.
fn boxes_overlap(a: &[f32; 4], b: &[f32; 4], margin: f32) -> bool {
    let a_expanded = [a[0] - margin, a[1] - margin, a[2] + margin, a[3] + margin];
    !(a_expanded[2] < b[0] || b[2] < a_expanded[0] || a_expanded[3] < b[1] || b[3] < a_expanded[1])
}

/// Calculate overlap area between two boxes.
fn overlap_area(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x_overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let y_overlap = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    x_overlap * y_overlap
}

/// Compute energy for a set of placements.
fn compute_energy(candidates: &[PlacementCandidate], config: &DeclutterConfig) -> f32 {
    let selected: Vec<_> = candidates.iter().filter(|c| c.selected).collect();

    let mut energy = 0.0;

    // Overlap penalty
    for i in 0..selected.len() {
        for j in (i + 1)..selected.len() {
            let overlap = overlap_area(&selected[i].bounds, &selected[j].bounds);
            energy += overlap * config.overlap_weight;
        }
    }

    // Priority bonus (negative energy for high-priority labels)
    for c in &selected {
        energy -= (c.priority as f32) * config.priority_weight;
    }

    // Distance penalty
    for c in candidates {
        if c.selected {
            energy += c.cost * config.distance_weight;
        }
    }

    energy
}

/// Greedy decluttering algorithm.
///
/// Places labels in priority order, skipping those that overlap
/// with already-placed labels.
fn apply_greedy_selection(candidates: &mut [PlacementCandidate], config: &DeclutterConfig) {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&left, &right| {
        candidates[right]
            .priority
            .cmp(&candidates[left].priority)
            .then(candidates[left].label_id.cmp(&candidates[right].label_id))
            .then(
                candidates[left]
                    .anchor_index
                    .cmp(&candidates[right].anchor_index),
            )
    });
    candidates
        .iter_mut()
        .for_each(|candidate| candidate.selected = false);
    let mut selected_labels = HashSet::new();
    let mut placed_bounds: Vec<[f32; 4]> = Vec::new();
    for index in order {
        let candidate = &candidates[index];
        if !candidate.visible || selected_labels.contains(&candidate.label_id) {
            continue;
        }
        if placed_bounds
            .iter()
            .any(|bounds| boxes_overlap(&candidate.bounds, bounds, config.margin))
        {
            continue;
        }
        candidates[index].selected = true;
        selected_labels.insert(candidates[index].label_id);
        placed_bounds.push(candidates[index].bounds);
    }
}

pub fn declutter_greedy(
    mut candidates: Vec<PlacementCandidate>,
    config: &DeclutterConfig,
) -> DeclutterResult {
    apply_greedy_selection(&mut candidates, config);

    let visible_labels = candidates
        .iter()
        .filter(|candidate| candidate.selected)
        .map(|candidate| candidate.label_id)
        .collect();
    let positions = candidates
        .iter()
        .filter(|candidate| candidate.selected)
        .map(|candidate| (candidate.label_id, candidate.position))
        .collect();

    let total_energy = compute_energy(&candidates, config);

    DeclutterResult {
        visible_labels,
        positions,
        total_energy,
        iterations: 1,
    }
}

/// Simulated annealing decluttering algorithm.
///
/// Uses stochastic optimization to find a near-optimal placement
/// that balances visibility of important labels with minimal overlap.
pub fn declutter_annealing(
    mut candidates: Vec<PlacementCandidate>,
    config: &DeclutterConfig,
) -> DeclutterResult {
    if candidates.is_empty() {
        return DeclutterResult {
            visible_labels: Vec::new(),
            positions: Vec::new(),
            total_energy: 0.0,
            iterations: 0,
        };
    }

    let mut rng = SimpleRng::new(config.seed);

    // Start with an identity-preserving greedy solution.
    apply_greedy_selection(&mut candidates, config);

    let mut current_energy = compute_energy(&candidates, config);
    let mut best_energy = current_energy;
    let mut best_selection: Vec<bool> = candidates.iter().map(|c| c.selected).collect();

    let mut temperature = config.initial_temperature;

    for _iteration in 0..config.max_iterations {
        // Pick a random candidate to toggle
        let idx = rng.next_usize(candidates.len());

        let previous_selection: Vec<bool> = candidates.iter().map(|c| c.selected).collect();
        if candidates[idx].selected {
            candidates[idx].selected = false;
        } else {
            if !candidates[idx].visible {
                continue;
            }
            let selected_label = candidates[idx].label_id;
            for candidate in &mut candidates {
                if candidate.label_id == selected_label {
                    candidate.selected = false;
                }
            }
            candidates[idx].selected = true;
        }

        let new_energy = compute_energy(&candidates, config);
        let delta = new_energy - current_energy;

        // Accept or reject the change
        let accept = if delta < 0.0 {
            true
        } else {
            let prob = (-delta / temperature).exp();
            rng.next_f32() < prob
        };

        if accept {
            current_energy = new_energy;
            if current_energy < best_energy {
                best_energy = current_energy;
                best_selection = candidates.iter().map(|c| c.selected).collect();
            }
        } else {
            for (candidate, selected) in candidates.iter_mut().zip(previous_selection) {
                candidate.selected = selected;
            }
        }

        // Cool down
        temperature *= config.cooling_rate;

        // Early termination if temperature is very low
        if temperature < 0.001 {
            break;
        }
    }

    // Apply best solution
    for (i, &selected) in best_selection.iter().enumerate() {
        candidates[i].selected = selected;
    }

    let visible_labels: Vec<u64> = candidates
        .iter()
        .filter(|c| c.selected)
        .map(|c| c.label_id)
        .collect();

    let positions: Vec<(u64, [f32; 2])> = candidates
        .iter()
        .filter(|c| c.selected)
        .map(|c| (c.label_id, c.position))
        .collect();

    DeclutterResult {
        visible_labels,
        positions,
        total_energy: best_energy,
        iterations: config.max_iterations,
    }
}

/// Declutter algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeclutterAlgorithm {
    /// Simple greedy algorithm (fast, decent results).
    #[default]
    Greedy,
    /// Simulated annealing (slower, better results).
    Annealing,
    /// Bounded-optimal branch-and-bound (deterministic, certified gap).
    Optimal,
}

/// Run decluttering with the specified algorithm.
pub fn declutter(
    candidates: Vec<PlacementCandidate>,
    config: &DeclutterConfig,
    algorithm: DeclutterAlgorithm,
) -> Result<DeclutterResult, super::optimal::CandidateError> {
    match algorithm {
        DeclutterAlgorithm::Greedy => Ok(declutter_greedy(candidates, config)),
        DeclutterAlgorithm::Annealing => Ok(declutter_annealing(candidates, config)),
        DeclutterAlgorithm::Optimal => super::optimal::declutter_optimal_result(candidates, config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: u64, x: f32, y: f32, priority: i32) -> PlacementCandidate {
        let size = 50.0;
        PlacementCandidate {
            label_id: id,
            anchor_index: 0,
            position: [x, y],
            bounds: [
                x - size / 2.0,
                y - size / 2.0,
                x + size / 2.0,
                y + size / 2.0,
            ],
            priority,
            cost: 0.0,
            selected: false,
            visible: true,
        }
    }

    #[test]
    fn test_greedy_non_overlapping() {
        let candidates = vec![
            make_candidate(1, 100.0, 100.0, 10),
            make_candidate(2, 200.0, 100.0, 5),
        ];
        let config = DeclutterConfig::default();
        let result = declutter_greedy(candidates, &config);
        assert_eq!(result.visible_labels.len(), 2);
    }

    #[test]
    fn test_greedy_overlapping_priority() {
        let candidates = vec![
            make_candidate(1, 100.0, 100.0, 10),
            make_candidate(2, 110.0, 100.0, 5), // Overlaps with 1
        ];
        let config = DeclutterConfig::default();
        let result = declutter_greedy(candidates, &config);
        assert_eq!(result.visible_labels.len(), 1);
        assert_eq!(result.visible_labels[0], 1); // Higher priority wins
    }

    #[test]
    fn test_boxes_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        assert!(boxes_overlap(&a, &b, 0.0));

        let c = [20.0, 20.0, 30.0, 30.0];
        assert!(!boxes_overlap(&a, &c, 0.0));
    }

    #[test]
    fn test_optimal_algorithm_dispatch() {
        let candidates = vec![
            make_candidate(1, 0.0, 0.0, 6),
            make_candidate(2, 30.0, 0.0, 10),
            make_candidate(3, 60.0, 0.0, 6),
        ];
        let config = DeclutterConfig {
            margin: 0.0,
            gap_tolerance: 0.0,
            ..DeclutterConfig::default()
        };
        let result = declutter(candidates, &config, DeclutterAlgorithm::Optimal)
            .expect("valid optimal dispatch");
        assert_eq!(result.visible_labels, vec![1, 3]);
    }

    #[test]
    fn test_greedy_selects_at_most_one_candidate_per_label() {
        let mut first = make_candidate(1, 0.0, 0.0, 10);
        first.anchor_index = 0;
        let mut second = make_candidate(1, 100.0, 0.0, 9);
        second.anchor_index = 1;
        let result = declutter_greedy(vec![second, first], &DeclutterConfig::default());
        assert_eq!(result.visible_labels, vec![1]);
        assert_eq!(result.positions, vec![(1, [0.0, 0.0])]);
    }

    #[test]
    fn test_annealing_preserves_candidate_identity_per_label() {
        let mut candidates = Vec::new();
        for (anchor_index, x) in [0.0, 100.0, 200.0].into_iter().enumerate() {
            let mut candidate = make_candidate(1, x, 0.0, 10 - anchor_index as i32);
            candidate.anchor_index = anchor_index as u32;
            candidates.push(candidate);
        }
        candidates.push(make_candidate(2, 400.0, 0.0, 5));
        let config = DeclutterConfig {
            max_iterations: 200,
            seed: 17,
            ..DeclutterConfig::default()
        };
        let result = declutter_annealing(candidates, &config);
        let label_one_count = result
            .visible_labels
            .iter()
            .filter(|&&label_id| label_id == 1)
            .count();
        assert!(label_one_count <= 1);
        assert_eq!(result.visible_labels.len(), result.positions.len());
    }
}
