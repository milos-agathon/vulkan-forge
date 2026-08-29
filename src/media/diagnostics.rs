use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationBreakdown {
    pub host_visible_bytes: u64,
    pub froxel_device_local_bytes: u64,
    pub density_device_local_bytes: u64,
    pub majorant_device_local_bytes: u64,
    pub staging_readback_bytes: u64,
}
