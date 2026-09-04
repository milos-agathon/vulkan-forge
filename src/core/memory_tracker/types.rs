/// Memory metrics returned to Python
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub buffer_count: u32,
    pub texture_count: u32,
    pub buffer_bytes: u64,
    pub texture_bytes: u64,
    pub host_visible_bytes: u64,
    pub total_bytes: u64,
    pub peak_host_visible_bytes: u64,
    pub peak_total_bytes: u64,
    pub limit_bytes: u64,
    pub within_budget: bool,
    pub utilization_ratio: f64,
    pub resident_tiles: u32,
    pub resident_tile_bytes: u64,
    pub resident_tiles_by_family: [u32; 4],
    pub resident_tile_bytes_by_family: [u64; 4],
    pub staging_bytes_in_flight: u64,
    pub staging_ring_count: u32,
    pub staging_buffer_size: u64,
    pub staging_buffer_stalls: u64,
    pub budget_policy: &'static str,
}

/// Statistics from defragmentation operation
#[derive(Debug, Clone, Default)]
pub struct DefragStats {
    /// Number of blocks moved during defragmentation
    pub blocks_moved: u32,
    /// Total bytes compacted
    pub bytes_compacted: u64,
    /// Time taken in milliseconds
    pub time_ms: f64,
    /// Fragmentation ratio before defrag (0.0-1.0)
    pub fragmentation_before: f32,
    /// Fragmentation ratio after defrag (0.0-1.0)
    pub fragmentation_after: f32,
}

/// Statistics for the memory pool system
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    /// Total bytes allocated from pools
    pub total_allocated: u64,
    /// Total bytes freed back to pools
    pub total_freed: u64,
    /// Current fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f32,
    /// Number of currently active blocks
    pub active_blocks: u32,
    /// Number of memory pools
    pub pool_count: u32,
    /// Size of largest free block
    pub largest_free_block: u64,
}
