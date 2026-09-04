//! Fence tracking system for staging buffer synchronization
//!
//! This module provides fence-based synchronization to ensure staging buffers
//! are not reused before GPU operations complete.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Queue, SubmissionIndex};

/// Tracks fence completion for staging buffers
pub struct FenceTracker {
    /// Device reference for fence operations
    device: Arc<Device>,
    /// Queue reference for fence submission
    queue: Arc<Queue>,
    /// Map of buffer index to fence value
    buffer_fences: HashMap<usize, u64>,
    /// Map of fence value to submission index
    fence_submissions: HashMap<u64, SubmissionIndex>,
    /// Next fence value to use
    next_fence_value: u64,
}

impl FenceTracker {
    /// Create a new fence tracker
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            buffer_fences: HashMap::new(),
            fence_submissions: HashMap::new(),
            next_fence_value: 1,
        }
    }

    /// Submit a fence for a specific buffer
    pub fn submit_fence(&mut self, buffer_index: usize, fence_value: u64) {
        // Create a command encoder to submit an empty command buffer with a fence
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("FenceTracker_Buffer_{}", buffer_index)),
            });

        // Submit the command buffer and get submission index
        let submission_index = self.queue.submit([encoder.finish()]);

        // Store the mapping
        self.buffer_fences.insert(buffer_index, fence_value);
        self.fence_submissions.insert(fence_value, submission_index);
    }

    /// Submit a fence with auto-generated value
    pub fn submit_fence_auto(&mut self, buffer_index: usize) -> u64 {
        let fence_value = self.next_fence_value;
        self.next_fence_value += 1;

        self.submit_fence(buffer_index, fence_value);
        fence_value
    }

    /// Check if a buffer is available for reuse
    pub fn is_buffer_available(&self, buffer_index: usize) -> bool {
        if let Some(&fence_value) = self.buffer_fences.get(&buffer_index) {
            if let Some(submission_index) = self.fence_submissions.get(&fence_value).cloned() {
                // Check if the submission has completed
                self.device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

                // For simplicity, we'll use a heuristic here
                // In a real implementation, you might want more sophisticated tracking
                true // Assume completed after polling
            } else {
                // No fence submitted yet, buffer is available
                true
            }
        } else {
            // No fence for this buffer yet, it's available
            true
        }
    }

    /// Wait for a specific fence to complete
    pub fn wait_for_fence(&self, fence_value: u64) {
        if let Some(submission_index) = self.fence_submissions.get(&fence_value).cloned() {
            self.device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));
        }
    }

    /// Wait for all pending fences to complete
    pub fn wait_for_all(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Clean up completed fences (remove from tracking)
    pub fn cleanup_completed(&mut self) {
        // Poll for completed submissions
        self.device.poll(wgpu::Maintain::Poll);

        // WGPU does not expose completion queries here; keep fences until explicit waits.
    }

    /// Get the number of pending fences
    pub fn pending_count(&self) -> usize {
        self.fence_submissions.len()
    }

    /// Clear all fence tracking (use with caution)
    pub fn clear_all(&mut self) {
        self.buffer_fences.clear();
        self.fence_submissions.clear();
    }
}

/// Helper for creating and managing fences during command buffer submission
pub struct FencedSubmission {
    /// The fence value associated with this submission
    pub fence_value: u64,
    /// The submission index returned by the queue
    pub submission_index: SubmissionIndex,
}

impl FencedSubmission {
    pub fn new(fence_value: u64, submission_index: SubmissionIndex) -> Self {
        Self {
            fence_value,
            submission_index,
        }
    }
}

/// Extension trait for Queue to simplify fence submission
pub trait QueueFenceExt {
    /// Submit command buffers with fence tracking
    fn submit_with_fence<I>(
        &self,
        command_buffers: I,
        fence_tracker: &mut FenceTracker,
        buffer_index: usize,
    ) -> FencedSubmission
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>;
}

impl QueueFenceExt for Queue {
    fn submit_with_fence<I>(
        &self,
        command_buffers: I,
        fence_tracker: &mut FenceTracker,
        buffer_index: usize,
    ) -> FencedSubmission
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        // Submit the command buffers
        let submission_index = self.submit(command_buffers);

        // Generate fence value and track it
        let fence_value = fence_tracker.submit_fence_auto(buffer_index);

        FencedSubmission::new(fence_value, submission_index)
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
    async fn test_fence_tracker_creation() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let tracker = FenceTracker::new(device, queue);

        assert_eq!(tracker.pending_count(), 0);
        assert!(tracker.is_buffer_available(0));
    }

    #[tokio::test]
    async fn test_fence_submission() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let mut tracker = FenceTracker::new(device, queue);

        let fence_value = tracker.submit_fence_auto(0);
        assert_eq!(fence_value, 1);
        assert_eq!(tracker.pending_count(), 1);
    }

    #[tokio::test]
    async fn test_buffer_availability() {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };
        let mut tracker = FenceTracker::new(device, queue);

        // Initially available
        assert!(tracker.is_buffer_available(0));

        // Submit fence
        tracker.submit_fence_auto(0);

        // Should still be available after polling (simplified implementation)
        assert!(tracker.is_buffer_available(0));
    }
}
