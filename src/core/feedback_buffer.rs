//! GPU feedback buffer system for virtual texture streaming
//!
//! This module provides GPU -> CPU communication for tile visibility feedback.
//! Terrain material VT writes feedback entries directly from the render shader,
//! then this buffer stages the data back to the CPU for residency updates.

use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use crate::core::tile_cache::TileId;
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Mutex;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue};

/// Largest number of slots the bounded feedback set will ever allocate.
///
/// TESSELLA win 1: the host-visible footprint of the feedback path must be a
/// function of this constant, never of the virtual texture's size. 65,536 slots
/// is 256 KiB of `MAP_READ` memory and is two orders of magnitude above the
/// distinct-page working set a 3840x2160 frame can produce (bounded by
/// `pixels / tile_area` summed over the mip chain).
pub const FEEDBACK_MAX_SLOTS: u32 = 1 << 16;

/// Legacy linear-probe limit retained for Rust source compatibility.
///
/// The bounded feedback path now appends with an atomic counter; this value is
/// not read. `TerrainVTUniforms.config3.y/z` carry page-table atlas dimensions.
#[deprecated(note = "bounded VT feedback no longer probes; this value is unused")]
pub const FEEDBACK_PROBE_LIMIT: u32 = 8;

/// Page-index layout the GPU keys were built from, used to invert
/// `terrain_vt_feedback_index` on readback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FeedbackLayout {
    /// Mip-0 page columns (`TerrainVTUniforms.config1.z`).
    pub base_pages_x: u32,
    /// Mip-0 page rows (`TerrainVTUniforms.config1.w`).
    pub base_pages_y: u32,
    /// Mip levels per logical material (`config2.x`).
    pub max_mip_levels: u32,
    /// Materials per family (`config2.y`).
    pub material_count: u32,
}

impl FeedbackLayout {
    /// Decode one non-empty slot back into the page it names.
    ///
    /// The GPU stores `page_index + 1` and nothing else, so this inversion is
    /// the only thing standing between a key and a tile request. It mirrors
    /// `terrain_vt_feedback_index` in `terrain_pbr_pom.wgsl` exactly.
    fn decode(&self, key: u32) -> Option<FeedbackEntry> {
        let index = key.checked_sub(1)?;
        let base_pages_x = self.base_pages_x.max(1);
        let base_pages_y = self.base_pages_y.max(1);
        let max_mip_levels = self.max_mip_levels.max(1);
        let material_count = self.material_count.max(1);
        let tile_x = index % base_pages_x;
        let rest = index / base_pages_x;
        let tile_y = rest % base_pages_y;
        let rest = rest / base_pages_y;
        let mip_level = rest % max_mip_levels;
        let logical_material = rest / max_mip_levels;
        if logical_material >= material_count.saturating_mul(3) {
            return None;
        }
        Some(FeedbackEntry {
            tile_x,
            tile_y,
            mip_level,
            frame_number: logical_material + 1,
        })
    }
}

/// GPU feedback buffer for collecting tile visibility information
pub struct FeedbackBuffer {
    /// GPU buffer for collecting feedback data from shaders
    feedback_buffer: TrackedBuffer,
    /// CPU-readable staging buffer for feedback readback
    readback_buffer: TrackedBuffer,
    pending_readback: Mutex<Option<Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    forced_not_ready_polls: AtomicU32,
    /// Append capacity in slots, excluding the two header words.
    capacity: u32,
    layout: FeedbackLayout,
    /// Samples the most recent readback could not admit. Non-zero means the
    /// frame's request set was incomplete -- surfaced through `vt_stats`, never
    /// swallowed.
    last_overflow: AtomicU32,
}

/// Feedback entry structure (matches GPU layout)
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct FeedbackEntry {
    /// Tile X coordinate
    pub tile_x: u32,
    /// Tile Y coordinate  
    pub tile_y: u32,
    /// Mip level
    pub mip_level: u32,
    /// Caller-defined payload. Terrain material VT uses
    /// `family_slot * material_count + material_index + 1`.
    pub frame_number: u32,
}

impl FeedbackBuffer {
    /// Create a bounded feedback set.
    ///
    /// `requested_slots` is rounded up to a power of two and clamped to
    /// [`FEEDBACK_MAX_SLOTS`], so the allocation cannot grow with the virtual
    /// texture. Two words precede the table: the append count and the explicit
    /// overflow count.
    pub fn new(
        device: &Device,
        requested_slots: u32,
        layout: FeedbackLayout,
    ) -> Result<Self, String> {
        let capacity = Self::capacity_for(requested_slots);
        // Append-count + overflow headers + `capacity` key slots, 4 bytes each.
        let buffer_size = (capacity as u64 + 2) * 4;

        // Create GPU feedback buffer
        let feedback_buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("FeedbackBuffer_GPU"),
                size: buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .map_err(|e| e.to_string())?;

        // Create CPU readback buffer
        let readback_buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("FeedbackBuffer_Readback"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .map_err(|e| e.to_string())?;

        Ok(Self {
            feedback_buffer,
            readback_buffer,
            pending_readback: Mutex::new(None),
            forced_not_ready_polls: AtomicU32::new(0),
            capacity,
            layout,
            last_overflow: AtomicU32::new(0),
        })
    }

    /// Round a requested slot count up to a power of two within the cap.
    pub fn capacity_for(requested_slots: u32) -> u32 {
        requested_slots
            .max(1)
            .min(FEEDBACK_MAX_SLOTS)
            .next_power_of_two()
            .min(FEEDBACK_MAX_SLOTS)
    }

    /// Append capacity in slots, excluding the two header words.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Samples the most recent readback could not admit.
    pub fn last_overflow(&self) -> u32 {
        self.last_overflow.load(Ordering::Acquire)
    }

    /// Clear feedback buffer for new frame
    pub fn clear(&self, encoder: &mut CommandEncoder) {
        // Clear feedback buffer by writing zeros
        encoder.clear_buffer(&self.feedback_buffer, 0, None);
    }

    /// Copy feedback data to readback buffer
    pub fn prepare_readback(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.feedback_buffer,
            0,
            &self.readback_buffer,
            0,
            self.feedback_buffer.size(),
        );
    }

    /// Whether a non-blocking readback map is currently in flight.
    pub fn has_pending_readback(&self) -> bool {
        self.pending_readback.lock().unwrap().is_some()
    }

    /// Delay the actual non-blocking map completion path for acceptance tests.
    pub fn force_not_ready_polls_for_test(&self, polls: u32) {
        self.forced_not_ready_polls.store(polls, Ordering::Release);
    }

    pub fn is_forced_not_ready_for_test(&self) -> bool {
        self.forced_not_ready_polls.load(Ordering::Acquire) > 0
    }

    fn start_readback_if_needed(&self) {
        let mut pending = self.pending_readback.lock().unwrap();
        if pending.is_some() {
            return;
        }

        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *pending = Some(receiver);
    }

    /// Read feedback data from GPU (async)
    pub async fn read_feedback_async(&self, device: &Device) -> Result<Vec<TileId>, String> {
        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive feedback data: {}", e))?
            .map_err(|e| format!("Failed to map feedback buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let entries = self.parse_feedback_tile_ids(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Read feedback data from GPU (blocking)
    pub fn read_feedback(&self, device: &Device, _queue: &Queue) -> Result<Vec<TileId>, String> {
        self.read_feedback_entries(device, _queue).map(|entries| {
            entries
                .into_iter()
                .map(|entry| TileId {
                    x: entry.tile_x,
                    y: entry.tile_y,
                    mip_level: entry.mip_level,
                })
                .collect()
        })
    }

    /// Read raw feedback entries from GPU (blocking)
    pub fn read_feedback_entries(
        &self,
        device: &Device,
        _queue: &Queue,
    ) -> Result<Vec<FeedbackEntry>, String> {
        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive feedback data: {}", e))?
            .map_err(|e| format!("Failed to map feedback buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let entries = self.parse_feedback_entries(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Read raw feedback entries, waiting for (or starting) the readback map.
    ///
    /// Unlike [`read_feedback_entries`], this cooperates with the
    /// non-blocking [`try_read_feedback_entries`] path: if an async map is
    /// already in flight it waits for that map instead of issuing a second
    /// (invalid) `map_async` on the same buffer.
    pub fn read_feedback_entries_blocking(
        &self,
        device: &Device,
    ) -> Result<Vec<FeedbackEntry>, String> {
        let pending = { self.pending_readback.lock().unwrap().take() };
        let receiver = match pending {
            Some(receiver) => receiver,
            None => {
                let buffer_slice = self.readback_buffer.slice(..);
                let (sender, receiver) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
                receiver
            }
        };

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive feedback data: {}", e))?
            .map_err(|e| format!("Failed to map feedback buffer: {:?}", e))?;

        let buffer_slice = self.readback_buffer.slice(..);
        let data = buffer_slice.get_mapped_range();
        let entries = self.parse_feedback_entries(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(entries)
    }

    /// Try to read raw feedback entries without blocking the frame.
    ///
    /// The first call starts a map operation and returns `Ok(None)`. Later calls
    /// poll the device once and return entries only after the map callback fires.
    pub fn try_read_feedback_entries(
        &self,
        device: &Device,
    ) -> Result<Option<Vec<FeedbackEntry>>, String> {
        self.start_readback_if_needed();
        if self
            .forced_not_ready_polls
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Ok(None);
        }
        device.poll(wgpu::Maintain::Poll);

        let map_result = {
            let mut pending = self.pending_readback.lock().unwrap();
            let Some(receiver) = pending.as_ref() else {
                return Ok(None);
            };
            match receiver.try_recv() {
                Ok(result) => {
                    *pending = None;
                    result
                }
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => {
                    *pending = None;
                    return Err("Feedback readback callback channel closed".to_string());
                }
            }
        };

        map_result.map_err(|e| format!("Failed to map feedback buffer: {:?}", e))?;

        let buffer_slice = self.readback_buffer.slice(..);
        let data = buffer_slice.get_mapped_range();
        let entries = self.parse_feedback_entries(&data);

        drop(data);
        self.readback_buffer.unmap();

        Ok(Some(entries))
    }

    /// Decode the bounded feedback set into deduplicated feedback entries.
    ///
    /// Word 0 is the append count, word 1 is the explicit overflow header, and
    /// every later non-zero word is a page key. The GPU uses a bounded append,
    /// so duplicate samples are expected; the `HashSet` is the authoritative
    /// CPU-side deduplication step.
    fn parse_feedback_entries(&self, data: &[u8]) -> Vec<FeedbackEntry> {
        let mut unique_entries = HashSet::new();

        let mut words = data.chunks_exact(4);
        let append_count = words
            .next()
            .and_then(|word| word.try_into().ok())
            .map(u32::from_le_bytes)
            .unwrap_or(0);
        let overflow = words
            .next()
            .and_then(|word| word.try_into().ok())
            .map(u32::from_le_bytes)
            .unwrap_or(0);
        self.last_overflow.store(overflow, Ordering::Release);
        if overflow > 0 {
            log::warn!(
                "feedback_buffer: {} surface samples exceeded the {}-slot feedback set; \
                 omitted samples must be regenerated by later frames before their pages can be admitted",
                overflow,
                self.capacity,
            );
        }

        // Only the slots reserved by the append counter are valid. The clear
        // command normally leaves the rest zero, but clamping here prevents a
        // stale or malformed tail from becoming a request on readback.
        let entry_limit = append_count.min(self.capacity) as usize;
        for (slot, word) in (&mut words).enumerate() {
            if slot >= entry_limit {
                continue;
            }
            let Ok(bytes) = <[u8; 4]>::try_from(word) else {
                continue;
            };
            let key = u32::from_le_bytes(bytes);
            if key == 0 {
                continue;
            }
            let Some(entry) = self.layout.decode(key) else {
                continue;
            };
            unique_entries.insert((
                entry.tile_x,
                entry.tile_y,
                entry.mip_level,
                entry.frame_number,
            ));
        }

        if !words.remainder().is_empty() {
            log::warn!(
                "feedback_buffer: discarded {} trailing bytes from GPU feedback stream",
                words.remainder().len()
            );
        }

        unique_entries
            .into_iter()
            .map(|(tile_x, tile_y, mip_level, frame_number)| FeedbackEntry {
                tile_x,
                tile_y,
                mip_level,
                frame_number,
            })
            .collect()
    }

    fn parse_feedback_tile_ids(&self, data: &[u8]) -> Vec<TileId> {
        self.parse_feedback_entries(data)
            .into_iter()
            .map(|entry| TileId {
                x: entry.tile_x,
                y: entry.tile_y,
                mip_level: entry.mip_level,
            })
            .collect()
    }

    /// Get feedback buffer for direct shader access
    pub fn buffer(&self) -> &Buffer {
        &self.feedback_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_entry_size() {
        assert_eq!(std::mem::size_of::<FeedbackEntry>(), 16);
    }

    #[test]
    fn test_feedback_entry_creation() {
        let entry = FeedbackEntry {
            tile_x: 10,
            tile_y: 20,
            mip_level: 2,
            frame_number: 100,
        };
        assert_eq!(entry.tile_x, 10);
        assert_eq!(entry.tile_y, 20);
        assert_eq!(entry.mip_level, 2);
        assert_eq!(entry.frame_number, 100);
    }

    fn test_layout() -> FeedbackLayout {
        FeedbackLayout {
            base_pages_x: 16,
            base_pages_y: 16,
            max_mip_levels: 4,
            material_count: 2,
        }
    }

    /// Mirror of `terrain_vt_feedback_index` in `terrain_pbr_pom.wgsl`.
    fn gpu_key(
        layout: FeedbackLayout,
        family: u32,
        material: u32,
        mip: u32,
        x: u32,
        y: u32,
    ) -> u32 {
        let logical_material = family * layout.material_count + material;
        (((logical_material * layout.max_mip_levels) + mip) * layout.base_pages_y + y)
            * layout.base_pages_x
            + x
            + 1
    }

    #[test]
    fn test_parse_empty_feedback_data() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            return;
        };

        let buffer = FeedbackBuffer::new(&device, 10, test_layout()).unwrap();

        let empty_data = vec![0u8; 0];
        let tiles = buffer.parse_feedback_tile_ids(&empty_data);
        assert!(tiles.is_empty());

        // Two headers + one empty slot.
        let zero_data = vec![0u8; 12];
        let tiles = buffer.parse_feedback_tile_ids(&zero_data);
        assert!(tiles.is_empty());
        assert_eq!(buffer.last_overflow(), 0);
    }

    #[test]
    fn test_parse_feedback_deduplicates_and_clamps_tail() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            return;
        };

        let layout = test_layout();
        let buffer = FeedbackBuffer::new(&device, 4, layout).unwrap();
        let mut bytes = 2u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let key = gpu_key(layout, 0, 0, 1, 3, 9).to_le_bytes();
        bytes.extend_from_slice(&key);
        bytes.extend_from_slice(&key);
        bytes.extend_from_slice(&gpu_key(layout, 0, 0, 1, 4, 9).to_le_bytes());
        bytes.extend_from_slice(&[0xAA, 0xBB]);

        let tiles = buffer.parse_feedback_tile_ids(&bytes);
        assert_eq!(tiles.len(), 1);
        assert!(tiles
            .iter()
            .any(|id| id.x == 3 && id.y == 9 && id.mip_level == 1));
    }

    /// The key alone must round-trip to the page it names, for every family and
    /// mip: it is the ONLY thing the GPU writes back now.
    #[test]
    fn feedback_keys_round_trip_to_their_page() {
        let layout = test_layout();
        for family in 0..3u32 {
            for material in 0..layout.material_count {
                for mip in 0..layout.max_mip_levels {
                    for (x, y) in [(0u32, 0u32), (1, 0), (0, 1), (15, 15), (7, 3)] {
                        let key = gpu_key(layout, family, material, mip, x, y);
                        let decoded = layout.decode(key).expect("key must decode");
                        assert_eq!((decoded.tile_x, decoded.tile_y), (x, y));
                        assert_eq!(decoded.mip_level, mip);
                        assert_eq!(
                            decoded.frame_number,
                            family * layout.material_count + material + 1,
                            "family {family} material {material}"
                        );
                    }
                }
            }
        }
        assert!(layout.decode(0).is_none(), "0 is the empty-slot marker");
    }

    /// The whole point of win 1: the allocation is a function of the CAPACITY,
    /// not of the virtual texture. A 2^18 x 2^18 virtual texture asks for
    /// 100,663,296 slots; the set must still cap at `FEEDBACK_MAX_SLOTS`.
    #[test]
    fn feedback_capacity_is_bounded_independently_of_the_virtual_texture() {
        assert_eq!(FeedbackBuffer::capacity_for(1), 1);
        assert_eq!(FeedbackBuffer::capacity_for(3000), 4096);
        assert_eq!(
            FeedbackBuffer::capacity_for(100_663_296),
            FEEDBACK_MAX_SLOTS
        );
        assert_eq!(FeedbackBuffer::capacity_for(u32::MAX), FEEDBACK_MAX_SLOTS);
        // Two header words + capacity keys, 4 bytes each: 256 KiB, versus the
        // 1,610,612,736 bytes the direct-mapped table used to demand.
        let bytes = (FEEDBACK_MAX_SLOTS as u64 + 2) * 4;
        assert!(bytes < 512 * 1024 * 1024 / 1000, "{bytes}");
        assert!(FEEDBACK_MAX_SLOTS.is_power_of_two());
    }

    /// Overflow is reported, never swallowed.
    #[test]
    fn feedback_overflow_header_is_surfaced() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            return;
        };
        let layout = test_layout();
        let buffer = FeedbackBuffer::new(&device, 4, layout).unwrap();
        let mut bytes = 1u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&17u32.to_le_bytes());
        bytes.extend_from_slice(&gpu_key(layout, 1, 0, 0, 2, 2).to_le_bytes());
        let entries = buffer.parse_feedback_entries(&bytes);
        assert_eq!(entries.len(), 1);
        assert_eq!(buffer.last_overflow(), 17);
    }
}
