//! Device-free per-family residency accounting for the terrain material VT.
//!
//! Terrain applies this feedback-driven policy to four family slots: one
//! material-runtime instance enables albedo, normal, and mask, while the
//! store-backed height mosaic owns a separate instance for height. Each
//! instance splits its budget across the families it enables, and eviction
//! pressure from one family never drains another family's resident set while
//! that family stays under its own budget (within-family LRU evicts first; the
//! owning cache capacity remains the instance-level backstop).
//!
//! Kept free of wgpu/PyO3 so the unit tests run under the curated cargo
//! feature set (which excludes `extension-module`).

use std::collections::VecDeque;

/// Number of terrain VT families (albedo, normal, mask, height).
pub(crate) const VT_FAMILY_COUNT: usize = 4;

/// Logical residency charge for one VT page. Every family is budgeted as
/// RGBA8 (4 bytes per texel), independent of a compressed device atlas.
pub(crate) const fn logical_resident_slot_bytes(slot_size: u32) -> u64 {
    slot_size as u64 * slot_size as u64 * 4
}

/// Validate a native material-VT budget after Python decoding. Python objects
/// are mutable and custom callers can bypass dataclass construction, so the
/// renderer must enforce this boundary again before allocating GPU resources.
pub(crate) fn validate_material_residency_budget(
    residency_budget_mb: f32,
    slot_size: u32,
    family_mask: u32,
    maximum_bytes: u64,
) -> Result<u64, String> {
    if !residency_budget_mb.is_finite() || residency_budget_mb <= 0.0 {
        return Err("terrain VT residency budget must be a positive finite value".to_string());
    }
    let budget_bytes = (f64::from(residency_budget_mb) * 1024.0 * 1024.0).floor() as u64;
    if budget_bytes > maximum_bytes {
        return Err(format!(
            "terrain VT residency budget {budget_bytes} bytes exceeds the 512 MiB memory limit"
        ));
    }
    let enabled_families = u64::from(family_mask.count_ones().max(1));
    let minimum_bytes = logical_resident_slot_bytes(slot_size) * enabled_families;
    if budget_bytes < minimum_bytes {
        return Err(format!(
            "terrain VT residency budget must hold at least one logical tile per enabled family ({minimum_bytes} bytes required)"
        ));
    }
    Ok(budget_bytes)
}

/// Validate that the physical atlas can reserve at least one slot for every
/// enabled family before the shared cache or per-family budgets are created.
pub(crate) fn validate_material_atlas_capacity(
    atlas_size: u32,
    slot_size: u32,
    family_mask: u32,
) -> Result<u32, String> {
    if slot_size == 0 || atlas_size < slot_size || !atlas_size.is_multiple_of(slot_size) {
        return Err(format!(
            "terrain VT atlas_size {atlas_size} must be divisible by non-zero slot_size {slot_size}"
        ));
    }
    let slots_axis = atlas_size / slot_size;
    let slots_total = slots_axis.saturating_mul(slots_axis);
    let enabled_families = family_mask.count_ones().max(1);
    if slots_total < enabled_families {
        return Err(format!(
            "terrain VT atlas must provide at least one physical slot per enabled family ({enabled_families} required, {slots_total} available)"
        ));
    }
    Ok(slots_total)
}

/// Fail closed on decoded family names. This is called by the native render
/// boundary because mutable/custom Python settings can bypass dataclass
/// construction-time validation.
pub(crate) fn validate_material_family_names(families: &[&str]) -> Result<(), String> {
    if families.is_empty() {
        return Err("enabled terrain VT requires at least one family".to_string());
    }
    let mut seen = Vec::new();
    for family in families {
        if !matches!(*family, "albedo" | "normal" | "mask") {
            return Err(format!(
                "terrain VT requested unsupported family '{family}'; supported families are albedo, normal, mask"
            ));
        }
        if seen.contains(family) {
            return Err(format!("terrain VT requested duplicate family '{family}'"));
        }
        seen.push(*family);
    }
    Ok(())
}

/// Admit already priority-sorted requests with round-robin family fairness.
/// Families with real shader feedback start each round before camera-only
/// families, so speculative/no-feedback work cannot consume the whole upload
/// budget ahead of demanded normal or mask pages.
pub(crate) fn fair_family_budget_admit<T>(
    mut queues: [VecDeque<(T, u64)>; VT_FAMILY_COUNT],
    feedback_family_mask: u32,
    budget_bytes: u64,
    admission_cursor: usize,
) -> Vec<T> {
    let mut family_order = (0..VT_FAMILY_COUNT)
        .map(|offset| (admission_cursor + offset) % VT_FAMILY_COUNT)
        .collect::<Vec<_>>();
    family_order.sort_by_key(|slot| feedback_family_mask & (1u32 << *slot as u32) == 0);
    let mut admitted = Vec::new();
    let mut spent = 0u64;
    loop {
        let mut inspected = false;
        for &slot in &family_order {
            let Some((request, cost)) = queues[slot].pop_front() else {
                continue;
            };
            inspected = true;
            if cost == 0 || spent.saturating_add(cost) <= budget_bytes {
                spent = spent.saturating_add(cost);
                admitted.push(request);
            }
        }
        if !inspected {
            break;
        }
    }
    admitted
}

/// Identity of one virtual-texture tile within a family/material/mip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TileKey {
    pub family_slot: u32,
    pub material_index: u32,
    pub x: u32,
    pub y: u32,
    pub mip_level: u32,
}

/// Residency snapshot for one material family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FamilyResidency {
    pub resident_tiles: u32,
    pub resident_bytes: u64,
    pub budget_bytes: u64,
}

/// Per-family residency budgets and LRU order.
pub(crate) struct FamilyResidencyTracker {
    tile_bytes: u64,
    total_budget_bytes: u64,
    families: [FamilyResidency; VT_FAMILY_COUNT],
    /// Access order per family; front = least recently used.
    lru: [VecDeque<TileKey>; VT_FAMILY_COUNT],
}

impl FamilyResidencyTracker {
    /// Split `total_budget_bytes` evenly across the families enabled in
    /// `family_mask` (bit `1 << slot`). Disabled families get a zero budget.
    pub fn new(total_budget_bytes: u64, family_mask: u32, tile_bytes: u64) -> Self {
        let enabled = (0..VT_FAMILY_COUNT)
            .filter(|slot| family_mask & (1u32 << slot) != 0)
            .count()
            .max(1) as u64;
        let per_family = total_budget_bytes / enabled;
        let mut families = [FamilyResidency::default(); VT_FAMILY_COUNT];
        for (slot, family) in families.iter_mut().enumerate() {
            if family_mask & (1u32 << slot) != 0 {
                family.budget_bytes = per_family;
            }
        }
        Self {
            tile_bytes: tile_bytes.max(1),
            total_budget_bytes,
            families,
            lru: Default::default(),
        }
    }

    fn slot_of(key: &TileKey) -> usize {
        key.family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize
    }

    /// Mark a resident tile as most recently used.
    pub fn note_access(&mut self, key: TileKey) {
        let slot = Self::slot_of(&key);
        if let Some(pos) = self.lru[slot].iter().position(|entry| *entry == key) {
            self.lru[slot].remove(pos);
            self.lru[slot].push_back(key);
        }
    }

    /// Record a newly resident tile (most recently used).
    pub fn on_insert(&mut self, key: TileKey) {
        let slot = Self::slot_of(&key);
        if self.lru[slot].iter().any(|entry| *entry == key) {
            self.note_access(key);
            return;
        }
        self.lru[slot].push_back(key);
        self.families[slot].resident_tiles += 1;
        self.families[slot].resident_bytes += self.tile_bytes;
    }

    /// Record an eviction (whether within-family or from the shared cache).
    pub fn on_evict(&mut self, key: &TileKey) {
        let slot = Self::slot_of(key);
        if let Some(pos) = self.lru[slot].iter().position(|entry| entry == key) {
            self.lru[slot].remove(pos);
            self.families[slot].resident_tiles =
                self.families[slot].resident_tiles.saturating_sub(1);
            self.families[slot].resident_bytes = self.families[slot]
                .resident_bytes
                .saturating_sub(self.tile_bytes);
        }
    }

    /// True when inserting one more tile would push the family over its own
    /// budget and it still has tiles that could be evicted.
    pub fn needs_eviction(&self, family_slot: u32) -> bool {
        let slot = family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize;
        let family = &self.families[slot];
        family.resident_tiles > 0 && family.resident_bytes + self.tile_bytes > family.budget_bytes
    }

    /// Least recently used resident tile of a family.
    pub fn lru_tile(&self, family_slot: u32) -> Option<TileKey> {
        let slot = family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize;
        self.lru[slot].front().copied()
    }

    /// Select a policy-owned victim before inserting `family_slot`.
    /// Within-family pressure always wins. If the aggregate budget is full,
    /// select from the family furthest above its fair share so the shared
    /// cache cannot bypass per-family accounting with an unrelated global LRU
    /// eviction.
    pub fn eviction_victim_for_insert(&self, family_slot: u32) -> Option<TileKey> {
        if self.needs_eviction(family_slot) {
            return self.lru_tile(family_slot);
        }
        if self.total_resident_bytes().saturating_add(self.tile_bytes) <= self.total_budget_bytes {
            return None;
        }
        self.families
            .iter()
            .enumerate()
            .filter(|(slot, family)| family.resident_tiles > 0 && !self.lru[*slot].is_empty())
            .max_by_key(|(_, family)| family.resident_bytes.saturating_sub(family.budget_bytes))
            .and_then(|(slot, _)| self.lru[slot].front().copied())
    }

    pub fn family(&self, family_slot: u32) -> FamilyResidency {
        self.families[family_slot.min(VT_FAMILY_COUNT as u32 - 1) as usize]
    }

    pub fn total_resident_bytes(&self) -> u64 {
        self.families.iter().map(|f| f.resident_bytes).sum()
    }
}

/// Decode a shader feedback payload (`logical_material + 1`) into
/// `(family_slot, material_index)`. Returns `None` for the zero sentinel and
/// for family slots outside the supported range.
pub(crate) fn decode_feedback_payload(payload: u32, material_count: u32) -> Option<(u32, u32)> {
    let encoded = payload.checked_sub(1)?;
    let material_count = material_count.max(1);
    let family_slot = encoded / material_count;
    let material_index = encoded % material_count;
    (family_slot < VT_FAMILY_COUNT as u32).then_some((family_slot, material_index))
}

/// Decode the full family/material/mip identity used by material-VT ingest.
/// Keeping mip validation beside payload demux makes independent per-family
/// mip feedback device-free and directly unit-testable.
pub(crate) fn decode_family_mip_feedback(
    payload: u32,
    mip_level: u32,
    material_count: u32,
    max_mip_levels: u32,
) -> Option<(u32, u32, u32)> {
    let (family_slot, material_index) = decode_feedback_payload(payload, material_count)?;
    (mip_level < max_mip_levels).then_some((family_slot, material_index, mip_level))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TILE: u64 = 256 * 256 * 4;

    fn key(family_slot: u32, x: u32, y: u32) -> TileKey {
        TileKey {
            family_slot,
            material_index: 0,
            x,
            y,
            mip_level: 0,
        }
    }

    #[test]
    fn budget_splits_evenly_across_enabled_families() {
        let total = 300 * TILE;
        let three = FamilyResidencyTracker::new(total, 0b111, TILE);
        assert_eq!(three.family(0).budget_bytes, total / 3);
        assert_eq!(three.family(1).budget_bytes, total / 3);
        assert_eq!(three.family(2).budget_bytes, total / 3);
        assert!(three.family(0).budget_bytes * 3 <= total);

        let two = FamilyResidencyTracker::new(total, 0b011, TILE);
        assert_eq!(two.family(0).budget_bytes, total / 2);
        assert_eq!(two.family(1).budget_bytes, total / 2);
        assert_eq!(two.family(2).budget_bytes, 0);

        let one = FamilyResidencyTracker::new(total, 0b010, TILE);
        assert_eq!(one.family(0).budget_bytes, 0);
        assert_eq!(one.family(1).budget_bytes, total);
        assert_eq!(one.family(2).budget_bytes, 0);
    }

    #[test]
    fn insert_and_evict_update_family_accounting() {
        let mut tracker = FamilyResidencyTracker::new(10 * TILE, 0b111, TILE);
        tracker.on_insert(key(1, 0, 0));
        tracker.on_insert(key(1, 1, 0));
        tracker.on_insert(key(2, 0, 0));
        assert_eq!(tracker.family(1).resident_tiles, 2);
        assert_eq!(tracker.family(1).resident_bytes, 2 * TILE);
        assert_eq!(tracker.family(2).resident_tiles, 1);
        assert_eq!(
            (0..VT_FAMILY_COUNT)
                .map(|slot| tracker.family(slot as u32).resident_tiles)
                .sum::<u32>(),
            3
        );
        assert_eq!(tracker.total_resident_bytes(), 3 * TILE);

        // Duplicate insert must not double-count.
        tracker.on_insert(key(1, 0, 0));
        assert_eq!(tracker.family(1).resident_tiles, 2);

        tracker.on_evict(&key(1, 0, 0));
        assert_eq!(tracker.family(1).resident_tiles, 1);
        assert_eq!(tracker.family(1).resident_bytes, TILE);
        // Evicting an unknown tile is a no-op.
        tracker.on_evict(&key(1, 9, 9));
        assert_eq!(tracker.family(1).resident_tiles, 1);
    }

    #[test]
    fn lru_order_follows_access_pattern() {
        let mut tracker = FamilyResidencyTracker::new(9 * TILE, 0b111, TILE);
        tracker.on_insert(key(0, 0, 0));
        tracker.on_insert(key(0, 1, 0));
        tracker.on_insert(key(0, 2, 0));
        assert_eq!(tracker.lru_tile(0), Some(key(0, 0, 0)));

        tracker.note_access(key(0, 0, 0));
        assert_eq!(tracker.lru_tile(0), Some(key(0, 1, 0)));

        tracker.on_evict(&key(0, 1, 0));
        assert_eq!(tracker.lru_tile(0), Some(key(0, 2, 0)));
    }

    #[test]
    fn needs_eviction_respects_per_family_budget_only() {
        // 6 tiles total budget -> 2 tiles per family.
        let mut tracker = FamilyResidencyTracker::new(6 * TILE, 0b111, TILE);
        tracker.on_insert(key(0, 0, 0));
        tracker.on_insert(key(0, 1, 0));
        // Family 0 is at budget; one more tile requires within-family eviction.
        assert!(tracker.needs_eviction(0));
        // Family 1 is empty and under budget: no eviction pressure, and it is
        // never asked to give up tiles on family 0's behalf.
        assert!(!tracker.needs_eviction(1));

        tracker.on_insert(key(1, 0, 0));
        assert!(!tracker.needs_eviction(1));
        assert!(tracker.needs_eviction(0));

        // Draining family 0 clears its pressure without touching family 1.
        tracker.on_evict(&key(0, 0, 0));
        tracker.on_evict(&key(0, 1, 0));
        assert!(!tracker.needs_eviction(0));
        assert_eq!(tracker.family(1).resident_tiles, 1);
    }

    #[test]
    fn within_family_eviction_loop_converges() {
        let mut tracker = FamilyResidencyTracker::new(3 * TILE, 0b111, TILE);
        // Budget = 1 tile per family.
        tracker.on_insert(key(2, 0, 0));
        assert!(tracker.needs_eviction(2));
        let victim = tracker.lru_tile(2).expect("family has a victim");
        tracker.on_evict(&victim);
        assert!(!tracker.needs_eviction(2));
        // Empty family never reports pressure (no infinite loops on tiny budgets).
        assert!(!tracker.needs_eviction(0));
        assert_eq!(tracker.lru_tile(0), None);
    }

    #[test]
    fn aggregate_pressure_selects_cross_family_victim() {
        // The odd tile is aggregate headroom that cannot be split evenly.
        let mut tracker = FamilyResidencyTracker::new(4 * TILE, 0b1111, TILE);
        tracker.on_insert(key(0, 0, 0));
        tracker.on_insert(key(1, 0, 0));
        tracker.on_insert(key(2, 0, 0));
        tracker.on_insert(key(2, 1, 0));
        assert_eq!(tracker.eviction_victim_for_insert(3), Some(key(2, 0, 0)));
    }

    #[test]
    fn logical_residency_charges_four_bytes_per_texel_for_every_family() {
        assert_eq!(logical_resident_slot_bytes(256), 256 * 256 * 4);
        assert_eq!(logical_resident_slot_bytes(128), 128 * 128 * 4);
    }

    #[test]
    fn native_budget_rejects_rounding_above_the_requested_limit() {
        let maximum = 512 * 1024 * 1024;
        let three_family_mask = 0b111;
        assert_eq!(
            validate_material_residency_budget(0.75, 256, three_family_mask, maximum),
            Ok(3 * 256 * 256 * 4),
        );
        assert!(
            validate_material_residency_budget(0.74, 256, three_family_mask, maximum)
                .unwrap_err()
                .contains("one logical tile per enabled family")
        );
        assert!(
            validate_material_residency_budget(f32::NAN, 256, three_family_mask, maximum).is_err()
        );
    }

    #[test]
    fn native_atlas_rejects_fewer_slots_than_enabled_families() {
        assert!(validate_material_atlas_capacity(256, 256, 0b111)
            .unwrap_err()
            .contains("one physical slot per enabled family"));
        assert_eq!(validate_material_atlas_capacity(512, 256, 0b111), Ok(4));
    }

    #[test]
    fn native_family_validation_rejects_mutated_unknowns_and_duplicates() {
        assert!(validate_material_family_names(&[]).is_err());
        assert!(validate_material_family_names(&["albedo", "bogus"]).is_err());
        assert!(validate_material_family_names(&["mask", "mask"]).is_err());
        assert!(validate_material_family_names(&["albedo", "normal", "mask"]).is_ok());
    }

    #[test]
    fn feedback_family_wins_tight_budget_and_round_robin_prevents_starvation() {
        let queues = [
            VecDeque::from([("albedo-camera-0", 1), ("albedo-camera-1", 1)]),
            VecDeque::from([("normal-feedback", 1)]),
            VecDeque::from([("mask-camera", 1)]),
            VecDeque::new(),
        ];
        assert_eq!(
            fair_family_budget_admit(queues.clone(), 0b010, 1, 0),
            vec!["normal-feedback"],
        );
        assert_eq!(
            fair_family_budget_admit(queues, 0b010, 4, 0),
            vec![
                "normal-feedback",
                "albedo-camera-0",
                "mask-camera",
                "albedo-camera-1",
            ],
        );
    }

    #[test]
    fn feedback_payload_demux_round_trips() {
        let material_count = 4;
        for family_slot in 0..VT_FAMILY_COUNT as u32 {
            for material_index in 0..material_count {
                let payload = family_slot * material_count + material_index + 1;
                assert_eq!(
                    decode_feedback_payload(payload, material_count),
                    Some((family_slot, material_index)),
                );
            }
        }
        // Zero is the "no feedback" sentinel.
        assert_eq!(decode_feedback_payload(0, material_count), None);
        // Payloads past the last family are rejected.
        let out_of_range = VT_FAMILY_COUNT as u32 * material_count + 1;
        assert_eq!(decode_feedback_payload(out_of_range, material_count), None);
        // material_count = 0 is clamped instead of dividing by zero.
        assert_eq!(decode_feedback_payload(1, 0), Some((0, 0)));
    }

    #[test]
    fn feedback_demux_preserves_independent_family_mips() {
        let material_count = 4;
        let records = [(0, 0, 3), (1, 2, 1), (2, 3, 2)];
        for (family_slot, material_index, mip_level) in records {
            let payload = family_slot * material_count + material_index + 1;
            assert_eq!(
                decode_family_mip_feedback(payload, mip_level, material_count, 4),
                Some((family_slot, material_index, mip_level)),
            );
        }
        assert_eq!(decode_family_mip_feedback(1, 4, material_count, 4), None);
    }

    #[test]
    fn persistent_cursor_prevents_multi_frame_feedback_starvation() {
        let mut winners = Vec::new();
        for cursor in 0..VT_FAMILY_COUNT {
            let queues = [
                VecDeque::from([("albedo-feedback", 1)]),
                VecDeque::from([("normal-feedback", 1)]),
                VecDeque::from([("mask-feedback", 1)]),
                VecDeque::new(),
            ];
            winners.extend(fair_family_budget_admit(queues, 0b111, 1, cursor));
        }
        assert_eq!(
            winners,
            vec![
                "albedo-feedback",
                "normal-feedback",
                "mask-feedback",
                "albedo-feedback"
            ]
        );
    }
}
