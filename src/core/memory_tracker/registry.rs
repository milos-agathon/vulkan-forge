use super::helpers::calculate_texture_size;
use super::MEMORY_BUDGET_LIMIT;
use crate::core::error::RenderError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use wgpu::TextureFormat;

const BUDGET_POLICY_ENFORCE: u8 = 0;
const BUDGET_POLICY_WARN: u8 = 1;

/// Global memory tracking registry for GPU resources.
pub struct ResourceRegistry {
    pub(super) buffer_count: AtomicU32,
    pub(super) texture_count: AtomicU32,
    pub(super) buffer_bytes: AtomicU64,
    pub(super) texture_bytes: AtomicU64,
    pub(super) host_visible_bytes: AtomicU64,
    // Exact subset owned by resource_tracker::ResourceHandle. Unlike the
    // public memory totals, these exclude estimate-only legacy bookkeeping and
    // therefore must exactly equal the allocation ledger.
    pub(super) ledger_host_visible_bytes: AtomicU64,
    pub(super) ledger_device_local_bytes: AtomicU64,
    pub(super) peak_host_visible_bytes: AtomicU64,
    pub(super) peak_total_bytes: AtomicU64,
    pub(super) resident_tiles: AtomicU32,
    pub(super) resident_tile_bytes: AtomicU64,
    pub(super) resident_tiles_by_family: [AtomicU32; 4],
    pub(super) resident_tile_bytes_by_family: [AtomicU64; 4],
    resident_owner_next: AtomicU64,
    resident_owners: Mutex<HashMap<u64, [(u32, u64); 4]>>,
    pub(super) staging_bytes_in_flight: AtomicU64,
    pub(super) staging_ring_count: AtomicU32,
    pub(super) staging_buffer_size: AtomicU64,
    pub(super) staging_buffer_stalls: AtomicU64,
    pub(super) budget_policy: AtomicU8,
    pub(super) budget_limit: u64,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            buffer_count: AtomicU32::new(0),
            texture_count: AtomicU32::new(0),
            buffer_bytes: AtomicU64::new(0),
            texture_bytes: AtomicU64::new(0),
            host_visible_bytes: AtomicU64::new(0),
            ledger_host_visible_bytes: AtomicU64::new(0),
            ledger_device_local_bytes: AtomicU64::new(0),
            peak_host_visible_bytes: AtomicU64::new(0),
            peak_total_bytes: AtomicU64::new(0),
            resident_tiles: AtomicU32::new(0),
            resident_tile_bytes: AtomicU64::new(0),
            resident_tiles_by_family: std::array::from_fn(|_| AtomicU32::new(0)),
            resident_tile_bytes_by_family: std::array::from_fn(|_| AtomicU64::new(0)),
            resident_owner_next: AtomicU64::new(1),
            resident_owners: Mutex::new(HashMap::new()),
            staging_bytes_in_flight: AtomicU64::new(0),
            staging_ring_count: AtomicU32::new(0),
            staging_buffer_size: AtomicU64::new(0),
            staging_buffer_stalls: AtomicU64::new(0),
            budget_policy: AtomicU8::new(BUDGET_POLICY_ENFORCE),
            budget_limit: MEMORY_BUDGET_LIMIT,
        }
    }

    pub fn set_budget_policy(&self, policy: &str) -> Result<&'static str, String> {
        let normalized = match policy {
            "enforce" => {
                self.budget_policy
                    .store(BUDGET_POLICY_ENFORCE, Ordering::Relaxed);
                "enforce"
            }
            "warn" => {
                self.budget_policy
                    .store(BUDGET_POLICY_WARN, Ordering::Relaxed);
                "warn"
            }
            _ => {
                return Err(format!(
                    "Unknown memory budget policy {policy:?}; expected 'enforce' or 'warn'"
                ));
            }
        };
        Ok(normalized)
    }

    pub fn get_budget_policy(&self) -> &'static str {
        match self.budget_policy.load(Ordering::Relaxed) {
            BUDGET_POLICY_WARN => "warn",
            _ => "enforce",
        }
    }

    /// Atomically enforce and reserve the process-wide host-visible budget,
    /// then register the buffer counters. Every authoritative host-visible
    /// allocation path uses this primitive so concurrent callers cannot each
    /// pass a stale read of the same remaining budget.
    pub fn track_buffer_allocation_labeled(
        &self,
        size: u64,
        is_host_visible: bool,
        label: &str,
    ) -> Result<(), RenderError> {
        if is_host_visible {
            let mut current = self.host_visible_bytes.load(Ordering::Acquire);
            loop {
                let next = current.checked_add(size).ok_or_else(|| {
                    RenderError::Budget(format!(
                        "Memory budget exceeded: allocation '{label}' requesting {size} bytes overflows aggregate host-visible accounting"
                    ))
                })?;
                let exceeds_budget = next > self.budget_limit;
                if exceeds_budget && self.get_budget_policy() == "enforce" {
                    let top5 = crate::core::resource_tracker::ledger_top_consumers_string(5);
                    return Err(RenderError::Budget(format!(
                        "Memory budget exceeded: allocation '{label}' requesting {size} bytes would exceed the 512 MiB host-visible limit (current: {current} bytes); top consumers: {top5}"
                    )));
                }
                match self.host_visible_bytes.compare_exchange_weak(
                    current,
                    next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        if exceeds_budget {
                            log::warn!(
                                "Memory budget exceeded: allocation '{}' requesting {} bytes exceeds the 512 MiB host-visible limit (current: {} bytes)",
                                label,
                                size,
                                current
                            );
                        }
                        self.record_peak_host_visible(next);
                        break;
                    }
                    Err(observed) => current = observed,
                }
            }
        }

        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.fetch_add(size, Ordering::Relaxed) + size;
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        self.record_peak_total(buffer_bytes.saturating_add(texture_bytes));

        Ok(())
    }

    /// Register a buffer under the same authoritative aggregate primitive.
    pub fn track_buffer_allocation(
        &self,
        size: u64,
        is_host_visible: bool,
    ) -> Result<(), RenderError> {
        self.track_buffer_allocation_labeled(size, is_host_visible, "unlabeled tracked buffer")
    }

    pub fn free_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        let _ = self
            .buffer_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            });
        let _ = self
            .buffer_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size))
            });

        if is_host_visible {
            let _ = self.host_visible_bytes.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| Some(current.saturating_sub(size)),
            );
        }
    }

    pub fn track_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);
        self.track_texture_allocation_bytes(size);
    }

    pub fn track_texture_allocation_bytes(&self, size: u64) {
        self.texture_count.fetch_add(1, Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.fetch_add(size, Ordering::Relaxed) + size;
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        self.record_peak_total(buffer_bytes.saturating_add(texture_bytes));
    }

    pub fn free_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);
        self.free_texture_allocation_bytes(size);
    }

    pub fn free_texture_allocation_bytes(&self, size: u64) {
        let _ = self
            .texture_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            });
        let _ = self
            .texture_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size))
            });
    }

    /// Record one allocation that also owns an allocation-ledger entry.
    pub fn track_ledger_allocation(&self, size: u64, is_host_visible: bool) {
        let counter = if is_host_visible {
            &self.ledger_host_visible_bytes
        } else {
            &self.ledger_device_local_bytes
        };
        counter.fetch_add(size, Ordering::Relaxed);
    }

    /// Remove one allocation that also owned an allocation-ledger entry.
    pub fn free_ledger_allocation(&self, size: u64, is_host_visible: bool) {
        let counter = if is_host_visible {
            &self.ledger_host_visible_bytes
        } else {
            &self.ledger_device_local_bytes
        };
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(size))
        });
    }

    /// Current registry totals for allocations that have ledger entries.
    pub fn ledger_totals(&self) -> (u64, u64) {
        (
            self.ledger_host_visible_bytes.load(Ordering::Relaxed),
            self.ledger_device_local_bytes.load(Ordering::Relaxed),
        )
    }

    pub fn set_resident_tiles(&self, count: u32, tile_bytes: u64) {
        self.resident_tiles.store(count, Ordering::Relaxed);
        self.resident_tile_bytes
            .store(tile_bytes, Ordering::Relaxed);
    }

    pub fn allocate_resident_owner(&self) -> u64 {
        self.resident_owner_next.fetch_add(1, Ordering::Relaxed)
    }

    /// Publish one owner's terrain VT family residency and recompute the true
    /// global per-family/aggregate footprint across every live renderer.
    pub fn set_resident_family_for_owner(
        &self,
        owner_id: u64,
        family_slot: usize,
        count: u32,
        tile_bytes: u64,
    ) {
        let slot = family_slot.min(self.resident_tiles_by_family.len() - 1);
        let mut owners = self
            .resident_owners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        owners.entry(owner_id).or_insert([(0, 0); 4])[slot] = (count, tile_bytes);
        if owners
            .get(&owner_id)
            .is_some_and(|families| families.iter().all(|value| *value == (0, 0)))
        {
            owners.remove(&owner_id);
        }
        self.refresh_resident_totals(&owners);
    }

    pub fn clear_resident_owner(&self, owner_id: u64) {
        let mut owners = self
            .resident_owners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        owners.remove(&owner_id);
        self.refresh_resident_totals(&owners);
    }

    fn refresh_resident_totals(&self, owners: &HashMap<u64, [(u32, u64); 4]>) {
        let mut family_tiles = [0u32; 4];
        let mut family_bytes = [0u64; 4];
        for families in owners.values() {
            for (slot, (count, bytes)) in families.iter().copied().enumerate() {
                family_tiles[slot] = family_tiles[slot].saturating_add(count);
                family_bytes[slot] = family_bytes[slot].saturating_add(bytes);
            }
        }
        for slot in 0..4 {
            self.resident_tiles_by_family[slot].store(family_tiles[slot], Ordering::Relaxed);
            self.resident_tile_bytes_by_family[slot].store(family_bytes[slot], Ordering::Relaxed);
        }
        self.set_resident_tiles(
            family_tiles.iter().copied().sum(),
            family_bytes.iter().copied().sum(),
        );
    }

    pub fn clear_resident_tiles(&self) {
        self.resident_owners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        for counter in &self.resident_tiles_by_family {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in &self.resident_tile_bytes_by_family {
            counter.store(0, Ordering::Relaxed);
        }
        self.set_resident_tiles(0, 0);
    }

    pub fn set_staging_stats(
        &self,
        bytes_in_flight: u64,
        ring_count: usize,
        buffer_size: u64,
        stalls: u64,
    ) {
        self.staging_bytes_in_flight
            .store(bytes_in_flight, Ordering::Relaxed);
        self.staging_ring_count
            .store(ring_count as u32, Ordering::Relaxed);
        self.staging_buffer_size
            .store(buffer_size, Ordering::Relaxed);
        self.staging_buffer_stalls.store(stalls, Ordering::Relaxed);
    }

    pub fn clear_staging_stats(&self) {
        self.set_staging_stats(0, 0, 0, 0);
    }

    fn record_peak_total(&self, value: u64) {
        let _ =
            self.peak_total_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.max(value))
                });
    }

    fn record_peak_host_visible(&self, value: u64) {
        let _ = self.peak_host_visible_bytes.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| Some(current.max(value)),
        );
    }
}

static GLOBAL_REGISTRY: std::sync::OnceLock<ResourceRegistry> = std::sync::OnceLock::new();

pub fn global_tracker() -> &'static ResourceRegistry {
    GLOBAL_REGISTRY.get_or_init(ResourceRegistry::new)
}
