mod helpers;
mod pool;
mod registry;
mod reporting;
#[cfg(test)]
mod tests;
mod types;

pub use helpers::{
    calculate_compressed_texture_size, calculate_texture_size, is_host_visible_usage,
};
pub use pool::{global_pools, init_global_pools, MemoryPoolManager, PoolBlock};
pub use registry::{global_tracker, ResourceRegistry};
pub use types::{DefragStats, MemoryMetrics, MemoryPoolStats};

pub(crate) const MEMORY_BUDGET_LIMIT: u64 = 512 * 1024 * 1024;
