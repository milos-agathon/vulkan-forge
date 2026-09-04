use super::{registry::ResourceRegistry, types::MemoryMetrics};
use std::sync::atomic::Ordering;

impl ResourceRegistry {
    pub fn get_metrics(&self) -> MemoryMetrics {
        let buffer_count = self.buffer_count.load(Ordering::Relaxed);
        let texture_count = self.texture_count.load(Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        let host_visible_bytes = self.host_visible_bytes.load(Ordering::Relaxed);
        let total_bytes = buffer_bytes + texture_bytes;
        let peak_host_visible_bytes = self.peak_host_visible_bytes.load(Ordering::Relaxed);
        let peak_total_bytes = self.peak_total_bytes.load(Ordering::Relaxed);
        let within_budget = host_visible_bytes <= self.budget_limit;
        let utilization_ratio = host_visible_bytes as f64 / self.budget_limit as f64;
        let resident_tiles = self.resident_tiles.load(Ordering::Relaxed);
        let resident_tile_bytes = self.resident_tile_bytes.load(Ordering::Relaxed);
        let resident_tiles_by_family =
            std::array::from_fn(|slot| self.resident_tiles_by_family[slot].load(Ordering::Relaxed));
        let resident_tile_bytes_by_family = std::array::from_fn(|slot| {
            self.resident_tile_bytes_by_family[slot].load(Ordering::Relaxed)
        });
        let staging_bytes_in_flight = self.staging_bytes_in_flight.load(Ordering::Relaxed);
        let staging_ring_count = self.staging_ring_count.load(Ordering::Relaxed);
        let staging_buffer_size = self.staging_buffer_size.load(Ordering::Relaxed);
        let staging_buffer_stalls = self.staging_buffer_stalls.load(Ordering::Relaxed);
        let budget_policy = self.get_budget_policy();

        MemoryMetrics {
            buffer_count,
            texture_count,
            buffer_bytes,
            texture_bytes,
            host_visible_bytes,
            total_bytes,
            peak_host_visible_bytes,
            peak_total_bytes,
            limit_bytes: self.budget_limit,
            within_budget,
            utilization_ratio,
            resident_tiles,
            resident_tile_bytes,
            resident_tiles_by_family,
            resident_tile_bytes_by_family,
            staging_bytes_in_flight,
            staging_ring_count,
            staging_buffer_size,
            staging_buffer_stalls,
            budget_policy,
        }
    }

    pub fn get_budget_limit(&self) -> u64 {
        self.budget_limit
    }

    pub fn check_budget(&self, additional_host_visible: u64) -> Result<(), String> {
        let current = self.host_visible_bytes.load(Ordering::Relaxed);
        if current.saturating_add(additional_host_visible) > self.budget_limit {
            let message = format!(
                "Memory budget exceeded: current {} bytes + requested {} bytes would exceed limit of {} bytes",
                current, additional_host_visible, self.budget_limit
            );
            if self.get_budget_policy() == "warn" {
                log::warn!("{message}");
                return Ok(());
            }
            return Err(message);
        }
        Ok(())
    }

    /// Labeled host-visible budget check used by the CENSOR tracked-buffer
    /// wrappers. On overage under the `enforce` policy this returns a
    /// [`RenderError::Budget`] naming the offending allocation and the ledger's
    /// top-5 consumers; under `warn` it logs and proceeds.
    pub fn check_budget_labeled(
        &self,
        additional_host_visible: u64,
        label: &str,
    ) -> Result<(), crate::core::error::RenderError> {
        let current = self.host_visible_bytes.load(Ordering::Relaxed);
        if current.saturating_add(additional_host_visible) > self.budget_limit {
            if self.get_budget_policy() == "warn" {
                log::warn!(
                    "Memory budget exceeded: allocation '{}' requesting {} bytes would exceed the 512 MiB host-visible limit (current: {} bytes)",
                    label, additional_host_visible, current
                );
                return Ok(());
            }
            let top5 = crate::core::resource_tracker::ledger_top_consumers_string(5);
            let message = format!(
                "Memory budget exceeded: allocation '{label}' requesting {additional_host_visible} bytes would exceed the 512 MiB host-visible limit (current: {current} bytes); top consumers: {top5}"
            );
            return Err(crate::core::error::RenderError::Budget(message));
        }
        Ok(())
    }
}
