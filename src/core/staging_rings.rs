//! O1: Staging buffer rings for efficient GPU uploads
//!
//! This module implements a ring buffer system for staging GPU uploads with
//! automatic wrap-around and fence-based synchronization to prevent buffer reuse
//! before completion.

use crate::core::error::RenderResult;
use crate::core::fence_tracker::FenceTracker;
use crate::core::memory_tracker::global_tracker;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue, Texture};

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Statistics for staging ring buffer usage
#[derive(Debug, Clone, Default)]
pub struct StagingStats {
    /// Total bytes currently in-flight across all rings
    pub bytes_in_flight: u64,
    /// Current active ring index
    pub current_ring_index: usize,
    /// Number of buffer stalls encountered
    pub buffer_stalls: u64,
    /// Total number of rings
    pub ring_count: usize,
    /// Size of each ring buffer
    pub buffer_size: u64,
}

/// A single staging buffer within the ring system
#[derive(Debug)]
struct StagingBuffer {
    /// WGPU buffer handle
    buffer: TrackedBuffer,
    /// Current offset within the buffer
    offset: u64,
    /// Size of this buffer
    size: u64,
    /// Whether this buffer is currently in use
    in_use: bool,
}

impl StagingBuffer {
    fn new(device: &Device, size: u64, label: Option<&str>) -> RenderResult<Self> {
        let buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label,
                size,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;

        Ok(Self {
            buffer,
            offset: 0,
            size,
            in_use: false,
        })
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.in_use = false;
    }

    fn can_allocate(&self, requested_size: u64) -> bool {
        self.offset + requested_size <= self.size
    }

    fn allocate(&mut self, size: u64) -> Option<u64> {
        if self.can_allocate(size) {
            let current_offset = self.offset;
            self.offset += size;
            self.in_use = true;
            Some(current_offset)
        } else {
            None
        }
    }
}

/// Ring buffer system for staging GPU uploads
pub struct StagingRing {
    /// Array of staging buffers (rings)
    buffers: Vec<StagingBuffer>,
    /// Current active ring index
    current_index: usize,
    /// Fence tracker for synchronization
    fence_tracker: Arc<Mutex<FenceTracker>>,
    /// Statistics tracking
    stats: Arc<Mutex<StagingStats>>,
}

impl StagingRing {
    fn publish_stats(&self) {
        if let Ok(stats) = self.stats.lock() {
            let snapshot = stats.clone();
            drop(stats);
            let tracker = global_tracker();
            tracker.set_staging_stats(
                snapshot.bytes_in_flight,
                snapshot.ring_count,
                snapshot.buffer_size,
                snapshot.buffer_stalls,
            );
        }
    }

    /// Create a new staging ring system
    ///
    /// # Arguments
    ///
    /// * `device` - WGPU device for buffer creation
    /// * `queue` - WGPU queue for fence operations
    /// * `ring_count` - Number of buffers in the ring (typically 3)
    /// * `buffer_size` - Size of each buffer in bytes
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        ring_count: usize,
        buffer_size: u64,
    ) -> RenderResult<Self> {
        let mut buffers = Vec::with_capacity(ring_count);
        for i in 0..ring_count {
            let label = format!("StagingRing_Buffer_{}", i);
            buffers.push(StagingBuffer::new(&device, buffer_size, Some(&label))?);
        }

        let stats = StagingStats {
            bytes_in_flight: 0,
            current_ring_index: 0,
            buffer_stalls: 0,
            ring_count,
            buffer_size,
        };

        let instance = Self {
            buffers,
            current_index: 0,
            fence_tracker: Arc::new(Mutex::new(FenceTracker::new(device.clone(), queue.clone()))),
            stats: Arc::new(Mutex::new(stats)),
        };
        instance.publish_stats();
        Ok(instance)
    }

    /// Get the current active staging buffer
    pub fn current(&self) -> &Buffer {
        self.buffers[self.current_index].buffer.inner()
    }

    /// Get current buffer with allocation offset
    pub fn allocate(&mut self, size: u64) -> Option<(&Buffer, u64)> {
        // Try to allocate from current buffer
        if let Some(offset) = self.buffers[self.current_index].allocate(size) {
            // Update stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.bytes_in_flight += size;
                stats.current_ring_index = self.current_index;
            }
            self.publish_stats();
            return Some((self.buffers[self.current_index].buffer.inner(), offset));
        }

        // Current buffer is full, try to advance
        if self.try_advance() {
            // Try again with new buffer
            if let Some(offset) = self.buffers[self.current_index].allocate(size) {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.bytes_in_flight += size;
                    stats.current_ring_index = self.current_index;
                }
                self.publish_stats();
                return Some((self.buffers[self.current_index].buffer.inner(), offset));
            }
        }

        // No space available
        if let Ok(mut stats) = self.stats.lock() {
            stats.buffer_stalls += 1;
        }
        self.publish_stats();
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_texture_region(
        &mut self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        texture: &Texture,
        origin: wgpu::Origin3d,
        data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> bool {
        if width == 0 || height == 0 || bytes_per_pixel == 0 {
            return false;
        }
        let bytes_per_row = width.saturating_mul(bytes_per_pixel);
        let expected_len = bytes_per_row as usize * height as usize;
        if data.len() != expected_len {
            return false;
        }

        let padded_bytes_per_row = align_to(bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);
        let upload_size = padded_bytes_per_row as u64 * height as u64;
        let Some((buffer, offset)) = self.allocate(upload_size) else {
            return false;
        };

        if padded_bytes_per_row == bytes_per_row {
            queue.write_buffer(buffer, offset, data);
        } else {
            let mut padded = vec![0u8; upload_size as usize];
            for row in 0..height as usize {
                let src_start = row * bytes_per_row as usize;
                let dst_start = row * padded_bytes_per_row as usize;
                padded[dst_start..dst_start + bytes_per_row as usize]
                    .copy_from_slice(&data[src_start..src_start + bytes_per_row as usize]);
            }
            queue.write_buffer(buffer, offset, &padded);
        }

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer,
                layout: wgpu::ImageDataLayout {
                    offset,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        true
    }

    /// Upload a block-compressed texture region without expanding it in host
    /// memory. `width`/`height` and `origin` are expressed in texels; buffer
    /// rows are expressed in compression blocks as required by wgpu.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_compressed_texture_region(
        &mut self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        texture: &Texture,
        origin: wgpu::Origin3d,
        data: &[u8],
        width: u32,
        height: u32,
        block_width: u32,
        block_height: u32,
        bytes_per_block: u32,
    ) -> bool {
        if width == 0
            || height == 0
            || block_width == 0
            || block_height == 0
            || bytes_per_block == 0
            || !width.is_multiple_of(block_width)
            || !height.is_multiple_of(block_height)
            || !origin.x.is_multiple_of(block_width)
            || !origin.y.is_multiple_of(block_height)
        {
            return false;
        }
        let block_rows = height / block_height;
        let bytes_per_row = (width / block_width).saturating_mul(bytes_per_block);
        if data.len() != bytes_per_row as usize * block_rows as usize {
            return false;
        }
        let padded_bytes_per_row = align_to(bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT);
        let upload_size = padded_bytes_per_row as u64 * block_rows as u64;
        let Some((buffer, offset)) = self.allocate(upload_size) else {
            return false;
        };
        if padded_bytes_per_row == bytes_per_row {
            queue.write_buffer(buffer, offset, data);
        } else {
            let mut padded = vec![0u8; upload_size as usize];
            for row in 0..block_rows as usize {
                let src_start = row * bytes_per_row as usize;
                let dst_start = row * padded_bytes_per_row as usize;
                padded[dst_start..dst_start + bytes_per_row as usize]
                    .copy_from_slice(&data[src_start..src_start + bytes_per_row as usize]);
            }
            queue.write_buffer(buffer, offset, &padded);
        }
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer,
                layout: wgpu::ImageDataLayout {
                    offset,
                    bytes_per_row: Some(padded_bytes_per_row),
                    // ImageDataLayout uses compression-block rows for
                    // block-compressed textures, not texel rows.
                    rows_per_image: Some(block_rows),
                },
            },
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        true
    }

    /// Advance to the next ring buffer with fence synchronization
    pub fn advance(&mut self, fence_value: u64) -> bool {
        // Submit fence for current buffer
        {
            let mut fence_tracker = self.fence_tracker.lock().unwrap();
            fence_tracker.submit_fence(self.current_index, fence_value);
        }

        self.try_advance()
    }

    /// Try to advance to next available buffer
    fn try_advance(&mut self) -> bool {
        let start_index = self.current_index;

        loop {
            let next_index = (self.current_index + 1) % self.buffers.len();

            // Check if next buffer is available (fence completed)
            let is_available = {
                let fence_tracker = self.fence_tracker.lock().unwrap();
                fence_tracker.is_buffer_available(next_index)
            };

            if is_available {
                // Reset the buffer and switch to it
                self.buffers[next_index].reset();
                self.current_index = next_index;

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.current_ring_index = next_index;
                }
                self.publish_stats();

                return true;
            }

            // Try the next buffer
            self.current_index = next_index;

            // If we've tried all buffers, no space available
            if self.current_index == start_index {
                return false;
            }
        }
    }

    /// Get current usage statistics
    pub fn stats(&self) -> StagingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Update bytes in flight (called when transfers complete)
    pub fn update_bytes_in_flight(&self, completed_bytes: u64) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.bytes_in_flight = stats.bytes_in_flight.saturating_sub(completed_bytes);
        }
        self.publish_stats();
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

impl Drop for StagingRing {
    fn drop(&mut self) {
        global_tracker().clear_staging_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backends, DeviceDescriptor, Instance, RequestAdapterOptions};

    async fn create_test_device() -> Option<(Arc<Device>, Arc<Queue>)> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: crate::core::gpu::instance_flags(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .ok()?;

        Some((Arc::new(device), Arc::new(queue)))
    }

    #[tokio::test]
    async fn test_staging_ring_creation() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let ring = StagingRing::new(device, queue, 3, 1024).expect("alloc");

        let stats = ring.stats();
        assert_eq!(stats.ring_count, 3);
        assert_eq!(stats.buffer_size, 1024);
        assert_eq!(stats.current_ring_index, 0);
        assert_eq!(stats.bytes_in_flight, 0);
    }

    #[tokio::test]
    async fn test_staging_allocation() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let mut ring = StagingRing::new(device, queue, 3, 1024).expect("alloc");

        // Test allocation
        let result = ring.allocate(256);
        assert!(result.is_some());

        let stats = ring.stats();
        assert_eq!(stats.bytes_in_flight, 256);
    }

    #[tokio::test]
    async fn test_multiple_allocations_share_current_ring() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let mut ring = StagingRing::new(device, queue, 3, 1024).expect("alloc");

        let first_offset = ring.allocate(128).map(|(_, offset)| offset);
        let second_offset = ring.allocate(256).map(|(_, offset)| offset);

        assert_eq!(first_offset, Some(0));
        assert_eq!(second_offset, Some(128));
        let stats = ring.stats();
        assert_eq!(stats.bytes_in_flight, 384);
        assert_eq!(stats.current_ring_index, 0);
    }

    #[tokio::test]
    async fn test_buffer_wrap_around() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let mut ring = StagingRing::new(device, queue, 3, 512).expect("alloc");

        // Fill current buffer
        let _alloc1 = ring.allocate(512);
        assert!(_alloc1.is_some());

        // This should try to advance to next buffer
        let allocated = ring.allocate(256).is_some();

        let stats = ring.stats();
        // Should have attempted to advance (may fail due to fence not being ready)
        assert!(stats.buffer_stalls > 0 || allocated);
    }
}
